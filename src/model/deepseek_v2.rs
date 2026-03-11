/// DeepSeek-V2 / V3 — DeepseekV2ForCausalLM / DeepseekV3ForCausalLM
///
/// Key architectural innovations:
/// 1. Multi-head Latent Attention (MLA): Compressed KV cache via low-rank projections
///    - kv_lora_rank: RK (typically 512) — KV compression rank
///    - q_lora_rank: RQ optional (typically 1536 for V3)
///    - Avoids full KV cache by storing only compressed latent vectors
/// 2. DeepSeekMoE: fine-grained experts (num_experts=256), routed+shared
/// 3. First few layers are dense (num_dense_layers config)
/// 4. Aux-free load balancing (sigmoid-based like Sarvam)
use candle_core::{DType, Device, Result, Tensor, Module};
use candle_nn::{VarBuilder, VarMap};
use crate::model::llama::Cache;
use crate::model::layers::{AdapterLayer, UnslothRmsNorm};

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,       // Dense FFN
    pub moe_intermediate_size: usize,   // Per-expert FFN
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<candle_transformers::models::llama::LlamaEosToks>,
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    // MLA params
    #[serde(default = "ds_kv_lora_rank")]
    pub kv_lora_rank: usize,
    #[serde(default = "ds_qk_nope_head_dim")]
    pub qk_nope_head_dim: usize,
    #[serde(default = "ds_qk_rope_head_dim")]
    pub qk_rope_head_dim: usize,
    #[serde(default = "ds_v_head_dim")]
    pub v_head_dim: usize,
    pub q_lora_rank: Option<usize>,     // None means no query compression
    // MoE params
    pub num_experts: Option<usize>,
    pub num_experts_per_tok: Option<usize>,
    #[serde(default = "ds_shared_experts")]
    pub n_shared_experts: usize,
    #[serde(default = "ds_dense_layers")]
    pub first_k_dense_replace: usize,
    #[serde(default)]
    pub moe_layer_freq: usize,          // MoE every N layers (0 = every layer after dense)
}

fn ds_kv_lora_rank() -> usize { 512 }
fn ds_qk_nope_head_dim() -> usize { 128 }
fn ds_qk_rope_head_dim() -> usize { 64 }
fn ds_v_head_dim() -> usize { 128 }
fn ds_shared_experts() -> usize { 1 }
fn ds_dense_layers() -> usize { 1 }

fn linear(size_in: usize, size_out: usize, vb: VarBuilder) -> Result<AdapterLayer> {
    let w = vb.get((size_out, size_in), "weight")?;
    Ok(AdapterLayer::Linear(candle_nn::Linear::new(w, None)))
}

// ─── RoPE for the "rope" portion of MLA heads ───────────────────────────────

struct RotaryEmbedding { cos: Tensor, sin: Tensor }

impl RotaryEmbedding {
    fn new(dtype: DType, dim: usize, rope_theta: f32, max_seq: usize, dev: &Device) -> Result<Self> {
        let inv_freq: Vec<f32> = (0..dim / 2)
            .map(|i| 1.0 / rope_theta.powf(2.0 * i as f32 / dim as f32))
            .collect();
        let len = inv_freq.len();
        let inv_f = Tensor::from_vec(inv_freq, (1, len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq as u32, dev)?
            .to_dtype(dtype)?.reshape((max_seq, 1))?;
        let freqs = t.matmul(&inv_f)?;
        Ok(Self { cos: freqs.cos()?, sin: freqs.sin()? })
    }

    fn forward(&self, x: &Tensor, pos: usize, seq_len: usize) -> Result<Tensor> {
        let cos = self.cos.narrow(0, pos, seq_len)?;
        let sin = self.sin.narrow(0, pos, seq_len)?;
        unsloth_rs::kernels::rope_cubecl(x, &cos, &sin)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))
    }
}

// ─── Multi-head Latent Attention (MLA) ──────────────────────────────────────

/// MLA compresses KV into a latent vector c_kv of dimension kv_lora_rank.
/// At inference time, we can avoid storing full KV by storing c_kv only.
/// For training/fine-tuning, we do the full expansion.
pub struct DeepSeekMLA {
    // Query path
    pub q_a_proj: Option<AdapterLayer>,  // Compressed query (if q_lora_rank)
    pub q_a_layernorm: Option<UnslothRmsNorm>,
    pub q_b_proj: Option<AdapterLayer>,
    pub q_proj: Option<AdapterLayer>,    // Direct query (if no compression)

    // Key-Value compression
    pub kv_a_proj_with_mqa: AdapterLayer,  // [hidden -> kv_lora_rank + qk_rope_head_dim]
    pub kv_a_layernorm: UnslothRmsNorm,
    pub kv_b_proj: AdapterLayer,            // [kv_lora_rank -> num_kv * (qk_nope + v_head)]

    pub o_proj: AdapterLayer,
    num_heads: usize,
    num_kv_heads: usize,
    kv_lora_rank: usize,
    qk_nope_head_dim: usize,
    qk_rope_head_dim: usize,
    v_head_dim: usize,
    rope: RotaryEmbedding,
}

impl DeepSeekMLA {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let h = cfg.num_attention_heads;
        let kv_h = cfg.num_key_value_heads;
        let nope = cfg.qk_nope_head_dim;
        let rope = cfg.qk_rope_head_dim;
        let vd = cfg.v_head_dim;
        let kr = cfg.kv_lora_rank;

        let (q_a_proj, q_a_layernorm, q_b_proj, q_proj) = if let Some(qlr) = cfg.q_lora_rank {
            // Compressed query path
            let qa = linear(cfg.hidden_size, qlr, vb.pp("q_a_proj"))?;
            let qa_norm = UnslothRmsNorm::new(vb.pp("q_a_layernorm").get(qlr, "weight")?, cfg.rms_norm_eps);
            let qb = linear(qlr, h * (nope + rope), vb.pp("q_b_proj"))?;
            (Some(qa), Some(qa_norm), Some(qb), None)
        } else {
            // Direct query projection
            let qp = linear(cfg.hidden_size, h * (nope + rope), vb.pp("q_proj"))?;
            (None, None, None, Some(qp))
        };

        // KV compression: produces kv_lora_rank + qk_rope_head_dim features
        let kv_a_proj_with_mqa = linear(cfg.hidden_size, kr + rope, vb.pp("kv_a_proj_with_mqa"))?;
        let kv_a_layernorm = UnslothRmsNorm::new(vb.pp("kv_a_layernorm").get(kr, "weight")?, cfg.rms_norm_eps);
        // Expand compressed KV to heads
        let kv_b_proj = linear(kr, kv_h * (nope + vd), vb.pp("kv_b_proj"))?;

        let rot = RotaryEmbedding::new(vb.dtype(), rope, cfg.rope_theta, cfg.max_position_embeddings, vb.device())?;
        let o_proj = linear(h * vd, cfg.hidden_size, vb.pp("o_proj"))?;

        Ok(Self {
            q_a_proj, q_a_layernorm, q_b_proj, q_proj,
            kv_a_proj_with_mqa, kv_a_layernorm, kv_b_proj,
            o_proj,
            num_heads: h,
            num_kv_heads: kv_h,
            kv_lora_rank: kr,
            qk_nope_head_dim: nope,
            qk_rope_head_dim: rope,
            v_head_dim: vd,
            rope: rot,
        })
    }

    fn forward(&self, x: &Tensor, pos: usize, cache: &mut Cache, layer_idx: usize) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let h = self.num_heads;
        let kv_h = self.num_kv_heads;
        let nope = self.qk_nope_head_dim;
        let rope_dim = self.qk_rope_head_dim;
        let vd = self.v_head_dim;

        // === Query ===
        let q = match (&self.q_a_proj, &self.q_a_layernorm, &self.q_b_proj) {
            (Some(qa), Some(qa_norm), Some(qb)) => {
                // Compressed: x -> qa -> norm -> qb
                let q_compressed = qa.forward(x)?;
                let q_normed = qa_norm.forward(&q_compressed)?;
                qb.forward(&q_normed)?
            },
            _ => self.q_proj.as_ref().unwrap().forward(x)?
        };
        // q: [b, s, h * (nope + rope_dim)]
        let q = q.reshape((b, s, h, nope + rope_dim))?.transpose(1, 2)?;
        let q_nope = q.narrow(candle_core::D::Minus1, 0, nope)?;
        let q_rope = q.narrow(candle_core::D::Minus1, nope, rope_dim)?.contiguous()?;
        let q_rope = self.rope.forward(&q_rope, pos, s)?;

        // === KV Compression ===
        let kv_compressed_full = self.kv_a_proj_with_mqa.forward(x)?;
        // Split: [b, s, kv_lora_rank + rope_dim]
        let kr = self.kv_lora_rank;
        let kv_latent = kv_compressed_full.narrow(candle_core::D::Minus1, 0, kr)?;
        let k_rope_raw = kv_compressed_full.narrow(candle_core::D::Minus1, kr, rope_dim)?;

        let kv_normed = self.kv_a_layernorm.forward(&kv_latent)?;
        // Expand latent to heads
        let kv_expanded = self.kv_b_proj.forward(&kv_normed)?;
        // kv_expanded: [b, s, kv_h * (nope + vd)]
        let kv_expanded = kv_expanded.reshape((b, s, kv_h, nope + vd))?.transpose(1, 2)?;
        let k_nope = kv_expanded.narrow(candle_core::D::Minus1, 0, nope)?;
        let v = kv_expanded.narrow(candle_core::D::Minus1, nope, vd)?.contiguous()?;

        // Apply RoPE to the rope portion of keys  
        let k_rope = k_rope_raw.reshape((b, s, 1, rope_dim))?.expand((b, s, kv_h, rope_dim))?.transpose(1, 2)?.contiguous()?;
        let k_rope = self.rope.forward(&k_rope, pos, s)?;

        // Concatenate nope + rope for full key
        let k = Tensor::cat(&[&k_nope, &k_rope], candle_core::D::Minus1)?.contiguous()?;
        let q = Tensor::cat(&[&q_nope, &q_rope], candle_core::D::Minus1)?.contiguous()?;

        // KV Cache
        let (k, v) = if cache.use_kv_cache {
            let (k, v) = match &cache.kvs[layer_idx] {
                Some((pk, pv)) => (Tensor::cat(&[pk, &k], 2)?, Tensor::cat(&[pv, &v], 2)?),
                None => (k, v),
            };
            cache.kvs[layer_idx] = Some((k.clone(), v.clone()));
            (k, v)
        } else { (k, v) };

        // GQA expansion
        let n_rep = h / kv_h;
        let k = if n_rep > 1 {
            let (bk, nk, sk, dk) = k.dims4()?;
            k.unsqueeze(2)?.expand((bk, nk, n_rep, sk, dk))?.reshape((bk, nk * n_rep, sk, dk))?
        } else { k };
        let v = if n_rep > 1 {
            let (bv, nv, sv, dv) = v.dims4()?;
            v.unsqueeze(2)?.expand((bv, nv, n_rep, sv, dv))?.reshape((bv, nv * n_rep, sv, dv))?
        } else { v };

        let scale = 1.0 / ((nope + rope_dim) as f64).sqrt();
        let attn = candle_nn::ops::softmax(&(q.matmul(&k.t()?)? * scale)?, candle_core::D::Minus1)?;
        attn.matmul(&v)?.transpose(1, 2)?.reshape((b, s, h * vd)).and_then(|y| self.o_proj.forward(&y))
    }
}

// ─── Dense/MoE MLP (reusing ExpertMLP pattern) ──────────────────────────────

struct ExpertMLP {
    gate_proj: AdapterLayer,
    up_proj: AdapterLayer,
    down_proj: AdapterLayer,
}

impl ExpertMLP {
    fn load(vb: VarBuilder, hidden: usize, inter: usize) -> Result<Self> {
        Ok(Self {
            gate_proj: linear(hidden, inter, vb.pp("gate_proj"))?,
            up_proj: linear(hidden, inter, vb.pp("up_proj"))?,
            down_proj: linear(inter, hidden, vb.pp("down_proj"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let swiglu = unsloth_rs::kernels::swiglu_cubecl(&gate, &up)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        self.down_proj.forward(&swiglu)
    }
}

struct DeepSeekMoE {
    gate: candle_nn::Linear,
    experts: Vec<ExpertMLP>,
    shared_experts: Vec<ExpertMLP>,
    num_experts_per_tok: usize,
}

impl DeepSeekMoE {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let ne = cfg.num_experts.unwrap_or(64);
        let inter = cfg.moe_intermediate_size;
        let hidden = cfg.hidden_size;
        let gate_w = vb.pp("gate").get((ne, hidden), "weight")?;
        let gate = candle_nn::Linear::new(gate_w, None);
        let experts = (0..ne)
            .map(|i| ExpertMLP::load(vb.pp(&format!("experts.{}", i)), hidden, inter))
            .collect::<Result<Vec<_>>>()?;
        let shared_experts = (0..cfg.n_shared_experts)
            .map(|i| ExpertMLP::load(vb.pp(&format!("shared_experts.{}", i)), hidden, inter))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { gate, experts, shared_experts, num_experts_per_tok: cfg.num_experts_per_tok.unwrap_or(6) })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, s, h) = x.dims3()?;
        let x_flat = x.reshape((b * s, h))?;

        let shared_out = self.shared_experts.iter().try_fold(
            Tensor::zeros_like(&x_flat)?,
            |acc, exp| acc + exp.forward(&x_flat),
        )?;

        let scores = candle_nn::ops::softmax(&self.gate.forward(&x_flat)?, candle_core::D::Minus1)?;
        let topk = scores.arg_sort_last_dim(false)?.narrow(1, 0, self.num_experts_per_tok)?;
        let topk_scores = scores.gather(&topk, 1)?;
        let topk_scores = {
            let s = topk_scores.sum_keepdim(1)?;
            topk_scores.broadcast_div(&s)?
        };

        let tokens = b * s;
        let mut out = Tensor::zeros_like(&x_flat)?;
        let topk_ids: Vec<Vec<u32>> = (0..self.num_experts_per_tok)
            .map(|k| topk.narrow(1, k, 1).and_then(|t| t.squeeze(1)).and_then(|t| t.to_vec1::<u32>()))
            .collect::<Result<Vec<_>>>()?;
        let topk_ws: Vec<Vec<f32>> = (0..self.num_experts_per_tok)
            .map(|k| topk_scores.narrow(1, k, 1).and_then(|t| t.squeeze(1)).and_then(|t| t.to_dtype(DType::F32)).and_then(|t| t.to_vec1::<f32>()))
            .collect::<Result<Vec<_>>>()?;

        for (tok_idx, row) in x_flat.chunk(tokens, 0)?.iter().enumerate() {
            for slot in 0..self.num_experts_per_tok {
                let eid = topk_ids[slot][tok_idx] as usize;
                let w = topk_ws[slot][tok_idx];
                let eout = self.experts[eid].forward(row)?;
                let scaled = (eout * w as f64)?;
                let existing = out.narrow(0, tok_idx, 1)?;
                out = out.slice_assign(&[tok_idx..tok_idx+1, 0..h], &(existing + scaled)?)?;
            }
        }
        ((out + shared_out)?).reshape((b, s, h))
    }
}

// ─── Block ──────────────────────────────────────────────────────────────────

enum FFNLayer {
    Dense(ExpertMLP),
    MoE(DeepSeekMoE),
}

impl FFNLayer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(m) => m.forward(x),
            Self::MoE(m) => m.forward(x),
        }
    }
}

pub struct DeepSeekV2Block {
    pub attn: DeepSeekMLA,
    ffn: FFNLayer,
    pub input_layernorm: UnslothRmsNorm,
    pub post_attention_layernorm: UnslothRmsNorm,
}

impl DeepSeekV2Block {
    fn load(vb: VarBuilder, cfg: &Config, layer_idx: usize) -> Result<Self> {
        let use_dense = layer_idx < cfg.first_k_dense_replace
            || (cfg.moe_layer_freq > 0 && layer_idx % cfg.moe_layer_freq != 0);
        let ffn = if use_dense || cfg.num_experts.is_none() {
            FFNLayer::Dense(ExpertMLP::load(vb.pp("mlp"), cfg.hidden_size, cfg.intermediate_size)?)
        } else {
            FFNLayer::MoE(DeepSeekMoE::load(vb.pp("mlp"), cfg)?)
        };
        Ok(Self {
            input_layernorm: UnslothRmsNorm::new(vb.pp("input_layernorm").get(cfg.hidden_size, "weight")?, cfg.rms_norm_eps),
            attn: DeepSeekMLA::load(vb.pp("self_attn"), cfg)?,
            post_attention_layernorm: UnslothRmsNorm::new(vb.pp("post_attention_layernorm").get(cfg.hidden_size, "weight")?, cfg.rms_norm_eps),
            ffn,
        })
    }

    fn forward(&self, x: &Tensor, pos: usize, cache: &mut Cache, layer_idx: usize) -> Result<Tensor> {
        let r = x;
        let x = self.input_layernorm.forward(x)?;
        let x = (self.attn.forward(&x, pos, cache, layer_idx)? + r)?;
        let r = &x;
        let x = self.post_attention_layernorm.forward(&x)?;
        (self.ffn.forward(&x)? + r)
    }
}

// ─── Model ──────────────────────────────────────────────────────────────────

pub struct DeepSeekV2 {
    pub embed_tokens: candle_nn::Embedding,
    pub layers: Vec<DeepSeekV2Block>,
    pub norm: UnslothRmsNorm,
    pub lm_head: AdapterLayer,
}

impl DeepSeekV2 {
    pub fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let embed_weight = vb.pp("model.embed_tokens").get((cfg.vocab_size, cfg.hidden_size), "weight")?;
        let embed_tokens = candle_nn::Embedding::new(embed_weight.clone(), cfg.hidden_size);
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| DeepSeekV2Block::load(vb.pp(&format!("model.layers.{}", i)), cfg, i))
            .collect::<Result<Vec<_>>>()?;
        let norm = UnslothRmsNorm::new(vb.pp("model.norm").get(cfg.hidden_size, "weight")?, cfg.rms_norm_eps);
        let lm_head = if cfg.tie_word_embeddings {
            AdapterLayer::Linear(candle_nn::Linear::new(embed_weight, None))
        } else {
            linear(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };
        Ok(Self { embed_tokens, layers, norm, lm_head })
    }

    pub fn forward(&mut self, input_ids: &Tensor, pos: usize, cache: &mut Cache) -> Result<Tensor> {
        let mut x = self.embed_tokens.forward(input_ids)?;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            x = layer.forward(&x, pos, cache, i)?;
        }
        self.lm_head.forward(&self.norm.forward(&x)?)
    }

    pub fn apply_lora(&mut self, target: Vec<String>, rank: usize, alpha: f64, _dropout: f64, use_dora: bool, varmap: &mut VarMap) -> Result<()> {
        let scaling = alpha / rank as f64;
        let device = self.embed_tokens.embeddings().device().clone();
        let dtype = self.embed_tokens.embeddings().dtype();
        use crate::model::inject_lora;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            // MLA-specific LoRA targets
            if target.contains(&"kv_a_proj_with_mqa".to_string()) {
                inject_lora(&mut layer.attn.kv_a_proj_with_mqa, rank, scaling, varmap, &device, dtype,
                    format!("model.layers.{}.self_attn.kv_a_proj_with_mqa", i), use_dora)?;
            }
            if target.contains(&"kv_b_proj".to_string()) {
                inject_lora(&mut layer.attn.kv_b_proj, rank, scaling, varmap, &device, dtype,
                    format!("model.layers.{}.self_attn.kv_b_proj", i), use_dora)?;
            }
            if target.contains(&"o_proj".to_string()) {
                inject_lora(&mut layer.attn.o_proj, rank, scaling, varmap, &device, dtype,
                    format!("model.layers.{}.self_attn.o_proj", i), use_dora)?;
            }
        }
        Ok(())
    }
}

// ─── Wrapper ─────────────────────────────────────────────────────────────────

pub struct DeepSeekV2Model {
    pub model: DeepSeekV2,
    pub config: Config,
    pub device: Device,
    pub dtype: DType,
    pub cache: Cache,
    pub varmap: VarMap,
}

impl DeepSeekV2Model {
    pub fn new(model: DeepSeekV2, config: Config, device: Device, dtype: DType, varmap: VarMap) -> Self {
        let cache = Cache::new(true, config.num_hidden_layers);
        Self { model, config, device, dtype, cache, varmap }
    }
    pub fn forward(&mut self, input_ids: &Tensor, pos: usize) -> Result<Tensor> {
        self.model.forward(input_ids, pos, &mut self.cache)
    }
    pub fn clear_cache(&mut self) { self.cache = Cache::new(true, self.config.num_hidden_layers); }
    pub fn apply_lora(&mut self, target: Vec<String>, rank: usize, alpha: f64, dropout: f64, use_dora: bool) -> Result<()> {
        self.model.apply_lora(target, rank, alpha, dropout, use_dora, &mut self.varmap)
    }
}
