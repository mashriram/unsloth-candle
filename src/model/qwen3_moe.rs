/// Qwen3-MoE model family (30B-A3B, 235B-A22B)
///
/// Architecture: Qwen3 dense blocks interleaved with MoE blocks
/// Key features:
/// - QK-Norm (same as Qwen3 dense)
/// - Shared expert + routed experts with top-k selection
/// - Configurable num_experts_per_tok, num_shared_experts
///
/// `Qwen3MoeForCausalLM`
use candle_core::{DType, Device, Result, Tensor, Module};
use candle_nn::{VarBuilder, VarMap, RmsNorm};
use crate::model::llama::Cache;
use crate::model::layers::{AdapterLayer, UnslothRmsNorm};

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub moe_intermediate_size: usize,
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
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    #[serde(default = "default_shared_experts")]
    pub num_shared_experts: usize,
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    // Layers that use dense MLP instead of MoE
    #[serde(default)]
    pub mlp_only_layers: Vec<usize>,
}

fn default_shared_experts() -> usize { 1 }
fn default_head_dim() -> usize { 128 }

fn linear(size_in: usize, size_out: usize, vb: VarBuilder) -> Result<AdapterLayer> {
    let w = vb.get((size_out, size_in), "weight")?;
    Ok(AdapterLayer::Linear(candle_nn::Linear::new(w, None)))
}

// ─── Head RMSNorm (QK-Norm) ─────────────────────────────────────────────────

struct HeadRmsNorm { norm: RmsNorm }

impl HeadRmsNorm {
    fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        Ok(Self { norm: RmsNorm::new(vb.get(dim, "weight")?, eps) })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, h, s, d) = x.dims4()?;
        let x2 = x.reshape((b * h * s, d))?;
        self.norm.forward(&x2)?.reshape((b, h, s, d))
    }
}

// ─── Rotary Embedding ───────────────────────────────────────────────────────

struct RotaryEmbedding { cos: Tensor, sin: Tensor }

impl RotaryEmbedding {
    fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let dim = cfg.head_dim;
        let inv_freq: Vec<f32> = (0..dim / 2)
            .map(|i| 1.0 / cfg.rope_theta.powf(2.0 * i as f32 / dim as f32))
            .collect();
        let len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, cfg.max_position_embeddings as u32, dev)?
            .to_dtype(dtype)?.reshape((cfg.max_position_embeddings, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self { cos: freqs.cos()?, sin: freqs.sin()? })
    }

    fn forward(&self, x: &Tensor, pos: usize, seq_len: usize) -> Result<Tensor> {
        let cos = self.cos.narrow(0, pos, seq_len)?;
        let sin = self.sin.narrow(0, pos, seq_len)?;
        unsloth_rs::kernels::rope_cubecl(x, &cos, &sin)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))
    }
}

// ─── Attention (same as Qwen3 dense) ────────────────────────────────────────

pub struct Qwen3MoeAttention {
    pub q_proj: AdapterLayer,
    pub k_proj: AdapterLayer,
    pub v_proj: AdapterLayer,
    pub o_proj: AdapterLayer,
    q_norm: HeadRmsNorm,
    k_norm: HeadRmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope: RotaryEmbedding,
}

impl Qwen3MoeAttention {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let hd = cfg.head_dim;
        let q_dim = cfg.num_attention_heads * hd;
        let kv_dim = cfg.num_key_value_heads * hd;
        Ok(Self {
            q_proj: linear(cfg.hidden_size, q_dim, vb.pp("q_proj"))?,
            k_proj: linear(cfg.hidden_size, kv_dim, vb.pp("k_proj"))?,
            v_proj: linear(cfg.hidden_size, kv_dim, vb.pp("v_proj"))?,
            o_proj: linear(q_dim, cfg.hidden_size, vb.pp("o_proj"))?,
            q_norm: HeadRmsNorm::new(hd, cfg.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: HeadRmsNorm::new(hd, cfg.rms_norm_eps, vb.pp("k_norm"))?,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: hd,
            rope: RotaryEmbedding::new(vb.dtype(), cfg, vb.device())?,
        })
    }

    fn forward(&self, x: &Tensor, pos: usize, cache: &mut Cache, layer_idx: usize) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let q = self.q_proj.forward(x)?.reshape((b, s, self.num_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let k = self.k_proj.forward(x)?.reshape((b, s, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let v = self.v_proj.forward(x)?.reshape((b, s, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;
        let q = self.rope.forward(&q, pos, s)?;
        let k = self.rope.forward(&k, pos, s)?;
        let (k, v) = if cache.use_kv_cache {
            let (k, v) = match &cache.kvs[layer_idx] {
                Some((pk, pv)) => (Tensor::cat(&[pk, &k], 2)?, Tensor::cat(&[pv, &v], 2)?),
                None => (k, v),
            };
            cache.kvs[layer_idx] = Some((k.clone(), v.clone()));
            (k, v)
        } else { (k, v) };
        let k = repeat_kv(k, self.num_heads / self.num_kv_heads)?;
        let v = repeat_kv(v, self.num_heads / self.num_kv_heads)?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn = candle_nn::ops::softmax(&(q.matmul(&k.t()?)? * scale)?, candle_core::D::Minus1)?;
        attn.matmul(&v)?.transpose(1, 2)?.reshape((b, s, self.num_heads * self.head_dim)).and_then(|y| self.o_proj.forward(&y))
    }
}

fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 { return Ok(x); }
    let (b, n_kv, s, d) = x.dims4()?;
    x.unsqueeze(2)?.expand((b, n_kv, n_rep, s, d))?.reshape((b, n_kv * n_rep, s, d))
}

// ─── Single Expert MLP ───────────────────────────────────────────────────────

struct ExpertMLP {
    gate_proj: AdapterLayer,
    up_proj: AdapterLayer,
    down_proj: AdapterLayer,
}

impl ExpertMLP {
    fn load(vb: VarBuilder, hidden: usize, intermediate: usize) -> Result<Self> {
        Ok(Self {
            gate_proj: linear(hidden, intermediate, vb.pp("gate_proj"))?,
            up_proj: linear(hidden, intermediate, vb.pp("up_proj"))?,
            down_proj: linear(intermediate, hidden, vb.pp("down_proj"))?,
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

// ─── Dense MLP (for non-MoE layers) ─────────────────────────────────────────

pub struct DenseMLP {
    pub gate_proj: AdapterLayer,
    pub up_proj: AdapterLayer,
    pub down_proj: AdapterLayer,
}

impl DenseMLP {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        Ok(Self {
            gate_proj: linear(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?,
            up_proj: linear(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?,
            down_proj: linear(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?,
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

// ─── MoE FFN ─────────────────────────────────────────────────────────────────

struct SparseMoE {
    gate: candle_nn::Linear,
    experts: Vec<ExpertMLP>,
    shared_experts: Vec<ExpertMLP>,
    num_experts_per_tok: usize,
}

impl SparseMoE {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let gate_weight = vb.pp("gate").get((cfg.num_experts, cfg.hidden_size), "weight")?;
        let gate = candle_nn::Linear::new(gate_weight, None);
        let experts = (0..cfg.num_experts)
            .map(|i| ExpertMLP::load(vb.pp(&format!("experts.{}", i)), cfg.hidden_size, cfg.moe_intermediate_size))
            .collect::<Result<Vec<_>>>()?;
        let shared_experts = (0..cfg.num_shared_experts)
            .map(|i| ExpertMLP::load(vb.pp(&format!("shared_expert_{}", i)), cfg.hidden_size, cfg.moe_intermediate_size))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { gate, experts, shared_experts, num_experts_per_tok: cfg.num_experts_per_tok })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, s, h) = x.dims3()?;
        let x_flat = x.reshape((b * s, h))?;

        // Compute shared expert output first
        let shared_out = self.shared_experts.iter().try_fold(
            Tensor::zeros_like(&x_flat)?,
            |acc, expert| (acc + expert.forward(&x_flat)?),
        )?;

        // Router logits and top-k selection
        let logits = self.gate.forward(&x_flat)?; // [b*s, num_experts]
        let probs = candle_nn::ops::softmax(&logits, candle_core::D::Minus1)?;

        // Top-k selection via argsort (descending)
        let topk_indices = logits.arg_sort_last_dim(false)?
            .narrow(1, 0, self.num_experts_per_tok)?;
        
        // Gather weights for selected experts
        let topk_probs = probs.gather(&topk_indices, 1)?;
        // Normalize
        let topk_probs_sum = topk_probs.sum_keepdim(1)?;
        let topk_probs = topk_probs.broadcast_div(&topk_probs_sum)?;

        // Dispatch tokens to experts
        let tokens = b * s;
        let mut routed_out = Tensor::zeros_like(&x_flat)?;

        for expert_slot in 0..self.num_experts_per_tok {
            let expert_ids = topk_indices.narrow(1, expert_slot, 1)?.squeeze(1)?;
            let expert_weights = topk_probs.narrow(1, expert_slot, 1)?;

            // Group tokens by expert
            let expert_ids_vec = expert_ids.to_vec1::<u32>()?;
            let mut expert_groups: Vec<Vec<usize>> = vec![vec![]; self.experts.len()];
            for (tok_idx, &exp_id) in expert_ids_vec.iter().enumerate() {
                expert_groups[exp_id as usize].push(tok_idx);
            }

            for (exp_id, tok_indices) in expert_groups.iter().enumerate() {
                if tok_indices.is_empty() { continue; }
                let tok_ids_u32: Vec<u32> = tok_indices.iter().map(|&i| i as u32).collect();
                let idx_tensor = Tensor::from_vec(tok_ids_u32, (tok_indices.len(),), x_flat.device())?;
                let subset = x_flat.index_select(&idx_tensor, 0)?;
                let expert_out = self.experts[exp_id].forward(&subset)?;
                // Scale by routing weight
                let weights = expert_weights.index_select(&idx_tensor, 0)?;
                let scaled = expert_out.broadcast_mul(&weights)?;
                // Scatter back
                for (out_pos, &tok_idx) in tok_indices.iter().enumerate() {
                    let row = scaled.narrow(0, out_pos, 1)?;
                    let existing = routed_out.narrow(0, tok_idx, 1)?;
                    routed_out = routed_out.slice_assign(&[tok_idx..tok_idx+1, 0..h], &(existing + row)?)?;
                }
            }
        }

        // Combine routed + shared
        let combined = (routed_out + shared_out)?;
        combined.reshape((b, s, h))
    }
}

// ─── Block ──────────────────────────────────────────────────────────────────

enum FFNLayer {
    MoE(SparseMoE),
    Dense(DenseMLP),
}

impl FFNLayer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::MoE(m) => m.forward(x),
            Self::Dense(d) => d.forward(x),
        }
    }
}

pub struct Qwen3MoeBlock {
    pub attn: Qwen3MoeAttention,
    ffn: FFNLayer,
    pub input_layernorm: UnslothRmsNorm,
    pub post_attention_layernorm: UnslothRmsNorm,
}

impl Qwen3MoeBlock {
    fn load(vb: VarBuilder, cfg: &Config, layer_idx: usize) -> Result<Self> {
        let use_dense = cfg.mlp_only_layers.contains(&layer_idx);
        let ffn = if use_dense {
            FFNLayer::Dense(DenseMLP::load(vb.pp("mlp"), cfg)?)
        } else {
            FFNLayer::MoE(SparseMoE::load(vb.pp("mlp"), cfg)?)
        };
        Ok(Self {
            input_layernorm: UnslothRmsNorm::new(vb.pp("input_layernorm").get(cfg.hidden_size, "weight")?, cfg.rms_norm_eps),
            attn: Qwen3MoeAttention::load(vb.pp("self_attn"), cfg)?,
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

pub struct Qwen3Moe {
    pub embed_tokens: candle_nn::Embedding,
    pub layers: Vec<Qwen3MoeBlock>,
    pub norm: UnslothRmsNorm,
    pub lm_head: AdapterLayer,
}

impl Qwen3Moe {
    pub fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let embed_weight = vb.pp("model.embed_tokens").get((cfg.vocab_size, cfg.hidden_size), "weight")?;
        let embed_tokens = candle_nn::Embedding::new(embed_weight.clone(), cfg.hidden_size);
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| Qwen3MoeBlock::load(vb.pp(&format!("model.layers.{}", i)), cfg, i))
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
            macro_rules! lora {
                ($module:expr, $name:expr) => {
                    if target.contains(&$name.to_string()) {
                        inject_lora(&mut $module, rank, scaling, varmap, &device, dtype,
                            format!("model.layers.{}.{}", i, $name), use_dora)?;
                    }
                };
            }
            lora!(layer.attn.q_proj, "self_attn.q_proj");
            lora!(layer.attn.k_proj, "self_attn.k_proj");
            lora!(layer.attn.v_proj, "self_attn.v_proj");
            lora!(layer.attn.o_proj, "self_attn.o_proj");
        }
        Ok(())
    }
}

// ─── Wrapper ─────────────────────────────────────────────────────────────────

pub struct Qwen3MoeModel {
    pub model: Qwen3Moe,
    pub config: Config,
    pub device: Device,
    pub dtype: DType,
    pub cache: Cache,
    pub varmap: VarMap,
}

impl Qwen3MoeModel {
    pub fn new(model: Qwen3Moe, config: Config, device: Device, dtype: DType, varmap: VarMap) -> Self {
        let cache = Cache::new(true, config.num_hidden_layers);
        Self { model, config, device, dtype, cache, varmap }
    }

    pub fn forward(&mut self, input_ids: &Tensor, pos: usize) -> Result<Tensor> {
        self.model.forward(input_ids, pos, &mut self.cache)
    }

    pub fn clear_cache(&mut self) {
        self.cache = Cache::new(true, self.config.num_hidden_layers);
    }

    pub fn apply_lora(&mut self, target: Vec<String>, rank: usize, alpha: f64, dropout: f64, use_dora: bool) -> Result<()> {
        self.model.apply_lora(target, rank, alpha, dropout, use_dora, &mut self.varmap)
    }
}
