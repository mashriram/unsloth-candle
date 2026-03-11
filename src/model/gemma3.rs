/// Gemma 3 model family (1B, 4B, 12B, 27B)
///
/// Key architectural differences from Gemma 2:
/// - Query-Key Normalization (QK-Norm) on attention heads
/// - Interleaved local (sliding window) and global attention
/// - 1B uses `GemmaForCausalLM`, larger models use `Gemma2ForCausalLM`
///   but with distinct config keys — this handles all variants correctly
/// - GeGLU activation in MLP (same as Gemma 2)
use candle_core::{DType, Device, Result, Tensor, Module};
use candle_nn::{VarBuilder, VarMap, RmsNorm, Activation};
use crate::model::llama::Cache;
use crate::model::layers::AdapterLayer;

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
    // Gemma 3: every N layers is a global attention layer
    #[serde(default = "default_sliding_window_pattern")]
    pub sliding_window_pattern: usize,
}

fn default_head_dim() -> usize { 256 }
fn default_sliding_window_pattern() -> usize { 6 } // every 6th layer is global

// ─── Gemma RMSNorm (weight * (1 + w)) ───────────────────────────────────────

struct GemmaRmsNorm {
    weight: Tensor,
    eps: f64,
}

impl GemmaRmsNorm {
    fn new(weight: Tensor, eps: f64) -> Self { Self { weight, eps } }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let x_f = x.to_dtype(DType::F32)?;
        let h = x_f.dim(candle_core::D::Minus1)?;
        let norm = ((x_f.sqr()?.sum_keepdim(candle_core::D::Minus1)? / h as f64)? + self.eps)?.sqrt()?;
        let x_norm = x_f.broadcast_div(&norm)?.to_dtype(dtype)?;
        // Gemma: x_norm * (1 + weight)
        x_norm.broadcast_mul(&(Tensor::ones_like(&self.weight)? + &self.weight)?)
    }
}

fn linear(size_in: usize, size_out: usize, vb: VarBuilder) -> Result<AdapterLayer> {
    let w = vb.get((size_out, size_in), "weight")?;
    let b = vb.get((size_out,), "bias").ok();
    Ok(AdapterLayer::Linear(candle_nn::Linear::new(w, b)))
}

// ─── Per-head RMSNorm for QK-Norm ───────────────────────────────────────────

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

pub struct Gemma3Attention {
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
    sliding_window: Option<usize>, // None = global attention
}

impl Gemma3Attention {
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
        let (b, s, _) = x.dims3()?;
        let q = self.q_proj.forward(x)?.reshape((b, s, self.num_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let k = self.k_proj.forward(x)?.reshape((b, s, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let v = self.v_proj.forward(x)?.reshape((b, s, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;

        // QK-Norm before RoPE
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

        // Sliding window for local attention layers
        let (k, v) = if let Some(window) = self.sliding_window {
            let total = k.dim(2)?;
            if total > window {
                (k.narrow(2, total - window, window)?, v.narrow(2, total - window, window)?)
            } else { (k, v) }
        } else { (k, v) };

        let k = repeat_kv(k, self.num_heads / self.num_kv_heads)?;
        let v = repeat_kv(v, self.num_heads / self.num_kv_heads)?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn = (q.matmul(&k.t()?)? * scale)?;

        let attn = if let Some(cap) = self.softcap {
            (attn / cap)?.tanh()?.affine(cap, 0.0)?
        } else { attn };

        let attn = candle_nn::ops::softmax(&attn, candle_core::D::Minus1)?;
        attn.matmul(&v)?.transpose(1, 2)?.reshape((b, s, self.num_heads * self.head_dim)).and_then(|y| self.o_proj.forward(&y))
    }
}

fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 { return Ok(x); }
    let (b, n_kv, s, d) = x.dims4()?;
    x.unsqueeze(2)?.expand((b, n_kv, n_rep, s, d))?.reshape((b, n_kv * n_rep, s, d))
}

// ─── MLP (GeGLU) ─────────────────────────────────────────────────────────────

pub struct Gemma3MLP {
    pub gate_proj: AdapterLayer,
    pub up_proj: AdapterLayer,
    pub down_proj: AdapterLayer,
}

impl Gemma3MLP {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        Ok(Self {
            gate_proj: linear(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?,
            up_proj: linear(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?,
            down_proj: linear(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // GeGLU: gelu(gate) * up
        let gate = self.gate_proj.forward(x)?.apply(&Activation::Gelu)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

// ─── Block ──────────────────────────────────────────────────────────────────

pub struct Gemma3Block {
    pub attn: Gemma3Attention,
    pub mlp: Gemma3MLP,
    pub input_layernorm: GemmaRmsNorm,
    pub post_attention_layernorm: GemmaRmsNorm,
    pub pre_feedforward_layernorm: GemmaRmsNorm,
    pub post_feedforward_layernorm: GemmaRmsNorm,
}

impl Gemma3Block {
    fn load(vb: VarBuilder, cfg: &Config, layer_idx: usize) -> Result<Self> {
        // Determine local vs global attention
        let is_local = (layer_idx + 1) % cfg.sliding_window_pattern != 0;
        Ok(Self {
            input_layernorm: GemmaRmsNorm::new(vb.pp("input_layernorm").get(cfg.hidden_size, "weight")?, cfg.rms_norm_eps),
            attn: Gemma3Attention::load(vb.pp("self_attn"), cfg, is_local)?,
            post_attention_layernorm: GemmaRmsNorm::new(vb.pp("post_attention_layernorm").get(cfg.hidden_size, "weight")?, cfg.rms_norm_eps),
            pre_feedforward_layernorm: GemmaRmsNorm::new(vb.pp("pre_feedforward_layernorm").get(cfg.hidden_size, "weight")?, cfg.rms_norm_eps),
            mlp: Gemma3MLP::load(vb.pp("mlp"), cfg)?,
            post_feedforward_layernorm: GemmaRmsNorm::new(vb.pp("post_feedforward_layernorm").get(cfg.hidden_size, "weight")?, cfg.rms_norm_eps),
        })
    }

    fn forward(&self, x: &Tensor, pos: usize, cache: &mut Cache, layer_idx: usize) -> Result<Tensor> {
        // Gemma 3 uses pre+post norm (double normalization per block)
        let r = x;
        let x = self.input_layernorm.forward(x)?;
        let x = self.attn.forward(&x, pos, cache, layer_idx)?;
        let x = (self.post_attention_layernorm.forward(&x)? + r)?;
        let r = &x;
        let x = self.pre_feedforward_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        (self.post_feedforward_layernorm.forward(&x)? + r)
    }
}

// ─── Model ──────────────────────────────────────────────────────────────────

pub struct Gemma3 {
    pub embed_tokens: candle_nn::Embedding,
    pub layers: Vec<Gemma3Block>,
    pub norm: GemmaRmsNorm,
    pub lm_head: AdapterLayer,
    pub final_softcap: Option<f64>,
}

impl Gemma3 {
    pub fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let embed_weight = vb.pp("model.embed_tokens").get((cfg.vocab_size, cfg.hidden_size), "weight")?;
        let embed_tokens = candle_nn::Embedding::new(embed_weight.clone(), cfg.hidden_size);
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| Gemma3Block::load(vb.pp(&format!("model.layers.{}", i)), cfg, i))
            .collect::<Result<Vec<_>>>()?;
        let norm = GemmaRmsNorm::new(vb.pp("model.norm").get(cfg.hidden_size, "weight")?, cfg.rms_norm_eps);
        let lm_head = if cfg.tie_word_embeddings {
            AdapterLayer::Linear(candle_nn::Linear::new(embed_weight, None))
        } else {
            linear(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };
        Ok(Self { embed_tokens, layers, norm, lm_head, final_softcap: cfg.final_logit_softcapping })
    }

    pub fn forward(&mut self, input_ids: &Tensor, pos: usize, cache: &mut Cache) -> Result<Tensor> {
        // Gemma 3: scale embeddings by sqrt(hidden_size)
        let mut x = self.embed_tokens.forward(input_ids)?;
        let scale = (x.dim(candle_core::D::Minus1)? as f64).sqrt();
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
}

// ─── Wrapper ─────────────────────────────────────────────────────────────────

pub struct Gemma3Model {
    pub model: Gemma3,
    pub config: Config,
    pub device: Device,
    pub dtype: DType,
    pub cache: Cache,
    pub varmap: VarMap,
}

impl Gemma3Model {
    pub fn new(model: Gemma3, config: Config, device: Device, dtype: DType, varmap: VarMap) -> Self {
        let cache = Cache::new(true, config.num_hidden_layers);
        Self { model, config, device, dtype, cache, varmap }
    }

    pub fn forward(&mut self, input_ids: &Tensor, pos: usize) -> Result<Tensor> {
        self.model.forward(input_ids, pos, &mut self.cache)
    }

    pub fn clear_cache(&mut self) {
        self.cache = Cache::new(true, self.config.num_hidden_layers);
    }

    pub fn apply_lora(&mut self, target: Vec<String>, rank: usize, alpha: f64, _dropout: f64, use_dora: bool) -> Result<()> {
        let scaling = alpha / rank as f64;
        let device = self.device.clone();
        let dtype = self.dtype;
        use crate::model::inject_lora;
        for (i, layer) in self.model.layers.iter_mut().enumerate() {
            macro_rules! lora {
                ($module:expr, $name:expr) => {
                    if target.contains(&$name.to_string()) {
                        inject_lora(&mut $module, rank, scaling, &mut self.varmap, &device, dtype,
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
