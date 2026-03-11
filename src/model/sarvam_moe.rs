/// Sarvam-30B — SarvamMoEForCausalLM
///
/// Custom MoE architecture optimized for Indian languages.
/// Key distinguishing features:
/// - 128 routed experts, top-6 selection with sigmoid routing (not softmax)
/// - Expert bias for load balancing
/// - QK-Norm on attention heads
/// - 1 shared expert per MoE block
/// - Dense FFN in first_k_dense_replace layers (default: layer 0)
/// - vocab_size = 262,144 (extended for subword coverage of Indian scripts)
/// - rope_theta = 8,000,000 (ultra-high for 128K+ context)
/// - GQA: 64 attention heads, 4 KV heads
/// - Hidden size 4096, 19 layers
use candle_core::{DType, Device, Result, Tensor, Module};
use candle_nn::{VarBuilder, VarMap, RmsNorm};
use crate::model::llama::Cache;
use crate::model::layers::{AdapterLayer, UnslothRmsNorm};

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,       // Dense FFN size
    pub moe_intermediate_size: usize,   // Routed expert size
    #[serde(default = "sarvam_shared_expert_intermediate_size")]
    pub moe_shared_expert_intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64, // Sarvam uses f64 precision here (8_000_000)
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<candle_transformers::models::llama::LlamaEosToks>,
    pub max_position_embeddings: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,     // top-k
    #[serde(default = "sarvam_shared_experts")]
    pub num_shared_experts: usize,
    #[serde(default = "sarvam_first_dense")]
    pub first_k_dense_replace: usize,   // First N layers use dense FFN
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    #[serde(default)]
    pub use_qk_norm: bool,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    // Routing options
    #[serde(default = "sarvam_score_fn")]
    pub score_function: String,         // "sigmoid" or "softmax"
    #[serde(default)]
    pub moe_router_enable_expert_bias: bool,
    #[serde(default = "sarvam_routed_scaling")]
    pub routed_scaling_factor: f64,
    #[serde(default = "sarvam_norm_topk")]
    pub norm_topk_prob: bool,
}

fn sarvam_shared_experts() -> usize { 1 }
fn sarvam_first_dense() -> usize { 1 }
fn default_head_dim() -> usize { 64 }
fn sarvam_shared_expert_intermediate_size() -> usize { 1024 }
fn sarvam_score_fn() -> String { "sigmoid".to_string() }
fn sarvam_routed_scaling() -> f64 { 2.5 }
fn sarvam_norm_topk() -> bool { true }

fn linear(size_in: usize, size_out: usize, vb: VarBuilder) -> Result<AdapterLayer> {
    let w = vb.get((size_out, size_in), "weight")?;
    Ok(AdapterLayer::Linear(candle_nn::Linear::new(w, None)))
}

// ─── Per-head QK-Norm ────────────────────────────────────────────────────────

struct HeadRmsNorm { norm: RmsNorm }

impl HeadRmsNorm {
    fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        Ok(Self { norm: RmsNorm::new(vb.get(dim, "weight")?, eps) })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, h, s, d) = x.dims4()?;
        self.norm.forward(&x.reshape((b * h * s, d))?)?.reshape((b, h, s, d))
    }
}

// ─── Rotary Embedding ───────────────────────────────────────────────────────

struct RotaryEmbedding { cos: Tensor, sin: Tensor }

impl RotaryEmbedding {
    fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let dim = cfg.head_dim;
        let theta = cfg.rope_theta as f32;
        let inv_freq: Vec<f32> = (0..dim / 2)
            .map(|i| 1.0 / theta.powf(2.0 * i as f32 / dim as f32))
            .collect();
        let len = inv_freq.len();
        let inv_f = Tensor::from_vec(inv_freq, (1, len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, cfg.max_position_embeddings as u32, dev)?
            .to_dtype(dtype)?.reshape((cfg.max_position_embeddings, 1))?;
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

// ─── Attention ──────────────────────────────────────────────────────────────

pub struct SarvamAttention {
    pub q_proj: AdapterLayer,
    pub k_proj: AdapterLayer,
    pub v_proj: AdapterLayer,
    pub o_proj: AdapterLayer,
    q_norm: Option<HeadRmsNorm>,
    k_norm: Option<HeadRmsNorm>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope: RotaryEmbedding,
}

impl SarvamAttention {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let hd = cfg.head_dim;
        let q_dim = cfg.num_attention_heads * hd;
        let kv_dim = cfg.num_key_value_heads * hd;
        let q_norm = if cfg.use_qk_norm {
            Some(HeadRmsNorm::new(hd, cfg.rms_norm_eps, vb.pp("q_norm"))?)
        } else { None };
        let k_norm = if cfg.use_qk_norm {
            Some(HeadRmsNorm::new(hd, cfg.rms_norm_eps, vb.pp("k_norm"))?)
        } else { None };
        Ok(Self {
            q_proj: linear(cfg.hidden_size, q_dim, vb.pp("q_proj"))?,
            k_proj: linear(cfg.hidden_size, kv_dim, vb.pp("k_proj"))?,
            v_proj: linear(cfg.hidden_size, kv_dim, vb.pp("v_proj"))?,
            o_proj: linear(q_dim, cfg.hidden_size, vb.pp("o_proj"))?,
            q_norm, k_norm,
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

        let q = if let Some(norm) = &self.q_norm { norm.forward(&q)? } else { q };
        let k = if let Some(norm) = &self.k_norm { norm.forward(&k)? } else { k };

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

        let n_rep = self.num_heads / self.num_kv_heads;
        let k = if n_rep > 1 {
            let (bk, nk, sk, dk) = k.dims4()?;
            k.unsqueeze(2)?.expand((bk, nk, n_rep, sk, dk))?.reshape((bk, nk * n_rep, sk, dk))?
        } else { k };
        let v = if n_rep > 1 {
            let (bv, nv, sv, dv) = v.dims4()?;
            v.unsqueeze(2)?.expand((bv, nv, n_rep, sv, dv))?.reshape((bv, nv * n_rep, sv, dv))?
        } else { v };

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn = candle_nn::ops::softmax(&(q.matmul(&k.t()?)? * scale)?, candle_core::D::Minus1)?;
        attn.matmul(&v)?.transpose(1, 2)?.reshape((b, s, self.num_heads * self.head_dim)).and_then(|y| self.o_proj.forward(&y))
    }
}

// ─── Dense MLP ───────────────────────────────────────────────────────────────

struct DenseMLP {
    gate_proj: AdapterLayer,
    up_proj: AdapterLayer,
    down_proj: AdapterLayer,
}

impl DenseMLP {
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

// ─── Sarvam MoE (Sigmoid routing with expert bias) ──────────────────────────

struct SarvamSparseMoE {
    gate: candle_nn::Linear,
    expert_bias: Option<Tensor>,        // Router bias for load balancing
    experts: Vec<DenseMLP>,
    shared_experts: Vec<DenseMLP>,
    num_experts_per_tok: usize,
    routed_scaling_factor: f64,
    use_sigmoid: bool,
    norm_topk_prob: bool,
}

impl SarvamSparseMoE {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let gate_w = vb.pp("gate").get((cfg.num_experts, cfg.hidden_size), "weight")?;
        let gate = candle_nn::Linear::new(gate_w, None);
        let expert_bias = if cfg.moe_router_enable_expert_bias {
            vb.pp("gate").get(cfg.num_experts, "e_score_correction_bias").ok()
        } else { None };

        let experts = (0..cfg.num_experts)
            .map(|i| DenseMLP::load(vb.pp(&format!("experts.{}", i)), cfg.hidden_size, cfg.moe_intermediate_size))
            .collect::<Result<Vec<_>>>()?;
        let shared_experts = (0..cfg.num_shared_experts)
            .map(|i| DenseMLP::load(vb.pp(&format!("shared_expert_{}", i)), cfg.hidden_size, cfg.moe_shared_expert_intermediate_size))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            gate,
            expert_bias,
            experts,
            shared_experts,
            num_experts_per_tok: cfg.num_experts_per_tok,
            routed_scaling_factor: cfg.routed_scaling_factor,
            use_sigmoid: cfg.score_function == "sigmoid",
            norm_topk_prob: cfg.norm_topk_prob,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, s, h) = x.dims3()?;
        let x_flat = x.reshape((b * s, h))?;
        let tokens = b * s;

        // Compute shared expert contributions
        let shared_out = self.shared_experts.iter().try_fold(
            Tensor::zeros_like(&x_flat)?,
            |acc, exp| acc + exp.forward(&x_flat),
        )?;

        // Router scores
        let logits = self.gate.forward(&x_flat)?; // [tokens, num_experts]

        let scores = if self.use_sigmoid {
            // Sarvam uses sigmoid scoring for independent expert probabilities
            let sig = candle_nn::ops::sigmoid(&logits)?;
            if let Some(bias) = &self.expert_bias {
                // Clone sig so we can use its shape AND move it into the addition
                let sig_shape = sig.shape().clone();
                (sig + bias.unsqueeze(0)?.broadcast_as(&sig_shape)?)?
            } else { sig }
        } else {
            candle_nn::ops::softmax(&logits, candle_core::D::Minus1)?
        };

        // Top-k selection
        let topk_indices = scores.arg_sort_last_dim(false)?
            .narrow(1, 0, self.num_experts_per_tok)?;
        let topk_scores = scores.gather(&topk_indices, 1)?;

        // Normalize top-k weights
        let topk_weights = if self.norm_topk_prob {
            let sum = topk_scores.sum_keepdim(1)?;
            topk_scores.broadcast_div(&sum)?
        } else {
            (topk_scores * self.routed_scaling_factor)?
        };

        // Dispatch: accumulate weighted expert outputs
        let mut routed_out = Tensor::zeros_like(&x_flat)?;
        let indices_vec: Vec<Vec<u32>> = (0..self.num_experts_per_tok)
            .map(|k| topk_indices.narrow(1, k, 1).and_then(|t| t.squeeze(1)).and_then(|t| t.to_vec1::<u32>()))
            .collect::<Result<Vec<_>>>()?;
        let weights_data: Vec<Vec<f32>> = (0..self.num_experts_per_tok)
            .map(|k| topk_weights.narrow(1, k, 1).and_then(|t| t.squeeze(1)).and_then(|t| t.to_dtype(DType::F32)).and_then(|t| t.to_vec1::<f32>()))
            .collect::<Result<Vec<_>>>()?;

        for (tok_idx, row) in x_flat.chunk(tokens, 0)?.iter().enumerate() {
            for slot in 0..self.num_experts_per_tok {
                let exp_id = indices_vec[slot][tok_idx] as usize;
                let weight = weights_data[slot][tok_idx];
                let expert_out = self.experts[exp_id].forward(row)?;
                let scaled = (expert_out * weight as f64)?;
                let existing = routed_out.narrow(0, tok_idx, 1)?;
                routed_out = routed_out.slice_assign(&[tok_idx..tok_idx+1, 0..h], &(existing + scaled)?)?;
            }
        }

        ((routed_out + shared_out)? ).reshape((b, s, h))
    }
}

// ─── Block ──────────────────────────────────────────────────────────────────

enum FFNLayer {
    Dense(DenseMLP),
    MoE(SarvamSparseMoE),
}

impl FFNLayer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(d) => d.forward(x),
            Self::MoE(m) => m.forward(x),
        }
    }
}

pub struct SarvamBlock {
    pub attn: SarvamAttention,
    ffn: FFNLayer,
    pub input_layernorm: UnslothRmsNorm,
    pub post_attention_layernorm: UnslothRmsNorm,
}

impl SarvamBlock {
    fn load(vb: VarBuilder, cfg: &Config, layer_idx: usize) -> Result<Self> {
        let is_dense = layer_idx < cfg.first_k_dense_replace;
        let ffn = if is_dense {
            FFNLayer::Dense(DenseMLP::load(vb.pp("mlp"), cfg.hidden_size, cfg.intermediate_size)?)
        } else {
            FFNLayer::MoE(SarvamSparseMoE::load(vb.pp("mlp"), cfg)?)
        };
        Ok(Self {
            input_layernorm: UnslothRmsNorm::new(vb.pp("input_layernorm").get(cfg.hidden_size, "weight")?, cfg.rms_norm_eps),
            attn: SarvamAttention::load(vb.pp("self_attn"), cfg)?,
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

pub struct SarvamMoe {
    pub embed_tokens: candle_nn::Embedding,
    pub layers: Vec<SarvamBlock>,
    pub norm: UnslothRmsNorm,
    pub lm_head: AdapterLayer,
}

impl SarvamMoe {
    pub fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let embed_weight = vb.pp("model.embed_tokens").get((cfg.vocab_size, cfg.hidden_size), "weight")?;
        let embed_tokens = candle_nn::Embedding::new(embed_weight.clone(), cfg.hidden_size);
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| SarvamBlock::load(vb.pp(&format!("model.layers.{}", i)), cfg, i))
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

pub struct SarvamMoeModel {
    pub model: SarvamMoe,
    pub config: Config,
    pub device: Device,
    pub dtype: DType,
    pub cache: Cache,
    pub varmap: VarMap,
}

impl SarvamMoeModel {
    pub fn new(model: SarvamMoe, config: Config, device: Device, dtype: DType, varmap: VarMap) -> Self {
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
