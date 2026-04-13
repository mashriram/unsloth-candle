/// Gemma 4 text model family (2B, 4B — text-only variants)
///
/// Gemma 4 is natively multimodal, but text-only paths follow the same
/// transformer architecture as Gemma 3 with these differences:
/// - Hybrid attention: alternating local sliding-window + global layers
/// - SwiGLU MLP (changed from GeGLU in Gemma 3)
/// - Larger sliding_window_pattern (every 5 layers global vs 6)
/// - model_type = "gemma4" / arch = "Gemma4ForCausalLM"
///
/// We reuse Gemma 3 internals and patch the config/arch registration.

use candle_core::{DType, Device, Result, Tensor, Module};
use candle_nn::{VarBuilder, VarMap, RmsNorm};
use crate::model::llama::Cache;
use crate::model::layers::{AdapterLayer, UnslothRmsNorm};

// ─── Config ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<candle_transformers::models::llama::LlamaEosToks>,
    pub max_position_embeddings: usize,
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    pub sliding_window: Option<usize>,
    pub attn_logit_softcapping: Option<f64>,
    pub final_logit_softcapping: Option<f64>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    /// Gemma 4: every N-th layer uses global attention (default 5)
    #[serde(default = "default_sliding_window_pattern")]
    pub sliding_window_pattern: usize,
    /// Gemma 4 uses SwiGLU instead of GeGLU
    #[serde(default)]
    pub use_swiglu: bool,
}

fn default_head_dim() -> usize { 256 }
fn default_sliding_window_pattern() -> usize { 5 }

// ─── Gemma RMSNorm (1 + weight) ──────────────────────────────────────────────

struct GemmaRmsNorm { weight: Tensor, eps: f64 }

impl GemmaRmsNorm {
    fn new(weight: Tensor, eps: f64) -> Self { Self { weight, eps } }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let xf = x.to_dtype(DType::F32)?;
        let h = xf.dim(candle_core::D::Minus1)?;
        let norm = ((xf.sqr()?.sum_keepdim(candle_core::D::Minus1)? / h as f64)? + self.eps)?.sqrt()?;
        let xn = xf.broadcast_div(&norm)?.to_dtype(dtype)?;
        xn.broadcast_mul(&(Tensor::ones_like(&self.weight)? + &self.weight)?)
    }
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

// ─── RoPE ────────────────────────────────────────────────────────────────────

struct RotaryEmbedding { cos: Tensor, sin: Tensor }

impl RotaryEmbedding {
    fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let dim = cfg.head_dim;
        let inv_freq: Vec<f32> = (0..dim / 2)
            .map(|i| 1.0 / cfg.rope_theta.powf(2.0 * i as f32 / dim as f32))
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
        candle_nn::rotary_emb::rope(x, &cos, &sin)
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn linear(si: usize, so: usize, vb: VarBuilder) -> Result<AdapterLayer> {
    let w = vb.get((so, si), "weight")?;
    Ok(AdapterLayer::Linear(candle_nn::Linear::new(w, None)))
}

fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 { return Ok(x); }
    let (b, n_kv, s, d) = x.dims4()?;
    x.unsqueeze(2)?.expand((b, n_kv, n_rep, s, d))?.reshape((b, n_kv * n_rep, s, d))
}

// ─── Attention ───────────────────────────────────────────────────────────────

pub struct Gemma4Attention {
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
    softcap: Option<f64>,
    sliding_window: Option<usize>,
}

impl Gemma4Attention {
    fn load(vb: VarBuilder, cfg: &Config, is_local: bool) -> Result<Self> {
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
            softcap: cfg.attn_logit_softcapping,
            sliding_window: if is_local { cfg.sliding_window } else { None },
        })
    }

    fn forward(&self, x: &Tensor, pos: usize, cache: &mut Cache, layer_idx: usize) -> Result<Tensor> {
        let (b, s, h_dim) = x.dims3()?;
        let q = self.q_proj.forward(x)?.reshape((b, s, self.num_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let k = self.k_proj.forward(x)?.reshape((b, s, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let v = self.v_proj.forward(x)?.reshape((b, s, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;

        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;
        let q = self.rope.forward(&q, pos, s)?;
        let k = self.rope.forward(&k, pos, s)?;

        let (k, v) = cache.append_and_fetch(layer_idx, &k, &v)?;

        let (k, v) = if let Some(window) = self.sliding_window {
            let total = k.dim(2)?;
            if total > window {
                (k.narrow(2, total - window, window)?, v.narrow(2, total - window, window)?)
            } else { (k, v) }
        } else { (k, v) };

        let k = repeat_kv(k, self.num_heads / self.num_kv_heads)?;
        let v = repeat_kv(v, self.num_heads / self.num_kv_heads)?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn = (q.matmul(&k.transpose(candle_core::D::Minus2, candle_core::D::Minus1)?)? * scale)?;
        let attn = if let Some(cap) = self.softcap {
            (attn / cap)?.tanh()?.affine(cap, 0.0)?
        } else { attn };
        let attn = candle_nn::ops::softmax(&attn, candle_core::D::Minus1)?;
        attn.matmul(&v)?.transpose(1, 2)?.reshape((b, s, h_dim)).and_then(|y| self.o_proj.forward(&y))
    }
}

// ─── MLP (SwiGLU — Gemma 4 uses SwiGLU unlike Gemma 3's GeGLU) ──────────────

pub struct Gemma4MLP {
    pub gate_proj: AdapterLayer,
    pub up_proj: AdapterLayer,
    pub down_proj: AdapterLayer,
}

impl Gemma4MLP {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        Ok(Self {
            gate_proj: linear(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?,
            up_proj: linear(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?,
            down_proj: linear(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // SwiGLU: silu(gate) * up
        let gate = self.gate_proj.forward(x)?.silu()?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

// ─── Block ───────────────────────────────────────────────────────────────────

pub struct Gemma4Block {
    pub attn: Gemma4Attention,
    pub mlp: Gemma4MLP,
    pub input_layernorm: GemmaRmsNorm,
    pub post_attention_layernorm: GemmaRmsNorm,
    pub pre_feedforward_layernorm: GemmaRmsNorm,
    pub post_feedforward_layernorm: GemmaRmsNorm,
}

impl Gemma4Block {
    fn load(vb: VarBuilder, cfg: &Config, layer_idx: usize) -> Result<Self> {
        let is_local = (layer_idx + 1) % cfg.sliding_window_pattern != 0;
        Ok(Self {
            input_layernorm: GemmaRmsNorm::new(vb.pp("input_layernorm").get(cfg.hidden_size, "weight")?, cfg.rms_norm_eps),
            attn: Gemma4Attention::load(vb.pp("self_attn"), cfg, is_local)?,
            post_attention_layernorm: GemmaRmsNorm::new(vb.pp("post_attention_layernorm").get(cfg.hidden_size, "weight")?, cfg.rms_norm_eps),
            pre_feedforward_layernorm: GemmaRmsNorm::new(vb.pp("pre_feedforward_layernorm").get(cfg.hidden_size, "weight")?, cfg.rms_norm_eps),
            mlp: Gemma4MLP::load(vb.pp("mlp"), cfg)?,
            post_feedforward_layernorm: GemmaRmsNorm::new(vb.pp("post_feedforward_layernorm").get(cfg.hidden_size, "weight")?, cfg.rms_norm_eps),
        })
    }

    fn forward(&self, x: &Tensor, pos: usize, cache: &mut Cache, layer_idx: usize) -> Result<Tensor> {
        let r = x;
        let h = self.input_layernorm.forward(x)?;
        let h = self.attn.forward(&h, pos, cache, layer_idx)?;
        let x = (self.post_attention_layernorm.forward(&h)? + r)?;
        let r = &x;
        let h = self.pre_feedforward_layernorm.forward(&x)?;
        let h = self.mlp.forward(&h)?;
        (self.post_feedforward_layernorm.forward(&h)? + r)
    }
}

// ─── Model ───────────────────────────────────────────────────────────────────

pub struct Gemma4 {
    pub embed_tokens: candle_nn::Embedding,
    pub layers: Vec<Gemma4Block>,
    pub norm: GemmaRmsNorm,
    pub lm_head: AdapterLayer,
    pub final_softcap: Option<f64>,
    hidden_size: usize,
}

impl Gemma4 {
    pub fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let embed_weight = vb.pp("model.embed_tokens").get((cfg.vocab_size, cfg.hidden_size), "weight")?;
        let embed_tokens = candle_nn::Embedding::new(embed_weight.clone(), cfg.hidden_size);
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| Gemma4Block::load(vb.pp(&format!("model.layers.{}", i)), cfg, i))
            .collect::<Result<Vec<_>>>()?;
        let norm = GemmaRmsNorm::new(vb.pp("model.norm").get(cfg.hidden_size, "weight")?, cfg.rms_norm_eps);
        let lm_head = if cfg.tie_word_embeddings {
            AdapterLayer::Linear(candle_nn::Linear::new(embed_weight, None))
        } else {
            linear(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };
        Ok(Self { embed_tokens, layers, norm, lm_head, final_softcap: cfg.final_logit_softcapping, hidden_size: cfg.hidden_size })
    }

    pub fn forward(&mut self, input_ids: &Tensor, pos: usize, cache: &mut Cache) -> Result<Tensor> {
        let mut x = self.embed_tokens.forward(input_ids)?;
        // Gemma normalisation scale
        let scale = (self.hidden_size as f64).sqrt();
        x = (x * scale)?;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            x = layer.forward(&x, pos, cache, i)?;
        }
        let x = self.norm.forward(&x)?;
        let logits = self.lm_head.forward(&x)?;
        if let Some(cap) = self.final_softcap {
            (logits / cap)?.tanh()?.affine(cap, 0.0)
        } else { Ok(logits) }
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
            lora!(layer.mlp.gate_proj, "mlp.gate_proj");
            lora!(layer.mlp.up_proj, "mlp.up_proj");
            lora!(layer.mlp.down_proj, "mlp.down_proj");
        }
        Ok(())
    }
}

// ─── Wrapper ─────────────────────────────────────────────────────────────────

pub struct Gemma4Model {
    pub model: Gemma4,
    pub config: Config,
    pub device: Device,
    pub dtype: DType,
    pub cache: Cache,
    pub varmap: VarMap,
}

impl Gemma4Model {
    pub fn new(model: Gemma4, config: Config, device: Device, dtype: DType, varmap: VarMap) -> Self {
        let cache = Cache::new(true, config.num_hidden_layers);
        Self { model, config, device, dtype, cache, varmap }
    }

    pub fn forward(&mut self, input_ids: &Tensor, pos: usize) -> Result<Tensor> {
        self.model.forward(input_ids, pos, &mut self.cache)
    }

    pub fn clear_cache(&mut self) {
        self.cache = Cache::new(true, self.config.num_hidden_layers);
    }

    pub fn configure_cache(&mut self, q: crate::core::cache::KVQuantization, rotor: bool) {
        self.cache.quantization = q;
        self.cache.use_rotor = rotor;
        self.clear_cache();
    }

    pub fn apply_lora(&mut self, target: Vec<String>, rank: usize, alpha: f64, dropout: f64, use_dora: bool) -> Result<()> {
        self.model.apply_lora(target, rank, alpha, dropout, use_dora, &mut self.varmap)
    }
}
