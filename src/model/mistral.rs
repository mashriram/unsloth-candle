/// Mistral / Ministral architecture
///
/// Separates Mistral from the Llama implementation to properly support:
/// - Sliding Window Attention (SWA)
/// - No tie_word_embeddings
/// - MistralForCausalLM architecture string
///
/// Compatible with: Mistral 7B v0.3, Mistral NeMo 12B, Ministral-3 3B/8B/14B
use candle_core::{DType, Device, Result, Tensor, Module};
use candle_nn::{VarBuilder, VarMap};
use crate::model::llama::Cache;
use crate::model::layers::{AdapterLayer, UnslothRmsNorm};

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
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub use_flash_attn: bool,
}

fn linear_no_bias(size_in: usize, size_out: usize, vb: VarBuilder) -> Result<AdapterLayer> {
    let weight = vb.get((size_out, size_in), "weight")?;
    Ok(AdapterLayer::Linear(candle_nn::Linear::new(weight, None)))
}

// ─── Rotary Embedding ───────────────────────────────────────────────────────

struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let dim = cfg.hidden_size / cfg.num_attention_heads;
        let max_seq = cfg.max_position_embeddings;
        let inv_freq: Vec<f32> = (0..dim / 2)
            .map(|i| 1.0 / cfg.rope_theta.powf(2.0 * i as f32 / dim as f32))
            .collect();
        let len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq, 1))?;
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

// ─── Attention ──────────────────────────────────────────────────────────────

pub struct MistralAttention {
    pub q_proj: AdapterLayer,
    pub k_proj: AdapterLayer,
    pub v_proj: AdapterLayer,
    pub o_proj: AdapterLayer,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope: RotaryEmbedding,
    sliding_window: Option<usize>,
}

impl MistralAttention {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let q_dim = cfg.num_attention_heads * head_dim;
        let kv_dim = cfg.num_key_value_heads * head_dim;
        Ok(Self {
            q_proj: linear_no_bias(cfg.hidden_size, q_dim, vb.pp("q_proj"))?,
            k_proj: linear_no_bias(cfg.hidden_size, kv_dim, vb.pp("k_proj"))?,
            v_proj: linear_no_bias(cfg.hidden_size, kv_dim, vb.pp("v_proj"))?,
            o_proj: linear_no_bias(q_dim, cfg.hidden_size, vb.pp("o_proj"))?,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim,
            rope: RotaryEmbedding::new(vb.dtype(), cfg, vb.device())?,
            sliding_window: cfg.sliding_window,
        })
    }

    fn forward(&self, x: &Tensor, pos: usize, cache: &mut Cache, layer_idx: usize) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let q = self.q_proj.forward(x)?.reshape((b, s, self.num_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let k = self.k_proj.forward(x)?.reshape((b, s, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let v = self.v_proj.forward(x)?.reshape((b, s, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;

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

        // Sliding window masking: trim old keys/values if window exceeded
        let (k, v) = if let Some(window) = self.sliding_window {
            let total_len = k.dim(2)?;
            if total_len > window {
                let start = total_len - window;
                (k.narrow(2, start, window)?, v.narrow(2, start, window)?)
            } else { (k, v) }
        } else { (k, v) };

        let k = self.repeat_kv(k)?;
        let v = self.repeat_kv(v)?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn = (q.matmul(&k.t()?)? * scale)?;
        let attn = candle_nn::ops::softmax(&attn, candle_core::D::Minus1)?;
        let y = attn.matmul(&v)?.transpose(1, 2)?.reshape((b, s, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&y)
    }

    fn repeat_kv(&self, x: Tensor) -> Result<Tensor> {
        let n_rep = self.num_heads / self.num_kv_heads;
        if n_rep == 1 { return Ok(x); }
        let (b, n_kv, s, d) = x.dims4()?;
        x.unsqueeze(2)?.expand((b, n_kv, n_rep, s, d))?.reshape((b, n_kv * n_rep, s, d))
    }
}

// ─── MLP (SwiGLU) ───────────────────────────────────────────────────────────

pub struct MistralMLP {
    pub gate_proj: AdapterLayer,
    pub up_proj: AdapterLayer,
    pub down_proj: AdapterLayer,
}

impl MistralMLP {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        Ok(Self {
            gate_proj: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?,
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

// ─── Block ──────────────────────────────────────────────────────────────────

pub struct MistralBlock {
    pub attn: MistralAttention,
    pub mlp: MistralMLP,
    pub input_layernorm: UnslothRmsNorm,
    pub post_attention_layernorm: UnslothRmsNorm,
}

impl MistralBlock {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        Ok(Self {
            input_layernorm: UnslothRmsNorm::new(vb.pp("input_layernorm").get(cfg.hidden_size, "weight")?, cfg.rms_norm_eps),
            attn: MistralAttention::load(vb.pp("self_attn"), cfg)?,
            post_attention_layernorm: UnslothRmsNorm::new(vb.pp("post_attention_layernorm").get(cfg.hidden_size, "weight")?, cfg.rms_norm_eps),
            mlp: MistralMLP::load(vb.pp("mlp"), cfg)?,
        })
    }

    fn forward(&self, x: &Tensor, pos: usize, cache: &mut Cache, layer_idx: usize) -> Result<Tensor> {
        let r = x;
        let x = self.input_layernorm.forward(x)?;
        let x = (self.attn.forward(&x, pos, cache, layer_idx)? + r)?;
        let r = &x;
        let x = self.post_attention_layernorm.forward(&x)?;
        (self.mlp.forward(&x)? + r)
    }
}

// ─── Model ──────────────────────────────────────────────────────────────────

pub struct Mistral {
    pub embed_tokens: candle_nn::Embedding,
    pub layers: Vec<MistralBlock>,
    pub norm: UnslothRmsNorm,
    pub lm_head: AdapterLayer,
}

impl Mistral {
    pub fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let embed_tokens = candle_nn::Embedding::new(
            vb.pp("model.embed_tokens").get((cfg.vocab_size, cfg.hidden_size), "weight")?, cfg.hidden_size);
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| MistralBlock::load(vb.pp(&format!("model.layers.{}", i)), cfg))
            .collect::<Result<Vec<_>>>()?;
        let norm = UnslothRmsNorm::new(vb.pp("model.norm").get(cfg.hidden_size, "weight")?, cfg.rms_norm_eps);
        let lm_head = if cfg.tie_word_embeddings {
            AdapterLayer::Linear(candle_nn::Linear::new(embed_tokens.embeddings().clone(), None))
        } else {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };
        Ok(Self { embed_tokens, layers, norm, lm_head })
    }

    pub fn forward(&mut self, input_ids: &Tensor, pos: usize, cache: &mut Cache) -> Result<Tensor> {
        let mut x = self.embed_tokens.forward(input_ids)?;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            x = layer.forward(&x, pos, cache, i)?;
        }
        let x = self.norm.forward(&x)?;
        self.lm_head.forward(&x)
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

pub struct MistralModel {
    pub model: Mistral,
    pub config: Config,
    pub device: Device,
    pub dtype: DType,
    pub cache: Cache,
    pub varmap: VarMap,
}

impl MistralModel {
    pub fn new(model: Mistral, config: Config, device: Device, dtype: DType, varmap: VarMap) -> Self {
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
