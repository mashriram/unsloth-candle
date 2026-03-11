/// Phi-4 — Phi4ForCausalLM
///
/// Phi-4 (14B) from Microsoft Research. 
/// Architecture very similar to Phi-3 with fused QKV projection but:
/// - Different hidden/intermediate sizes (14B scale)
/// - Phi4ForCausalLM architecture string
/// - No mandatory sliding window (uses global attention)
/// - Same SwiGLU MLP via fused gate_up_proj
use candle_core::{DType, Device, Result, Tensor, Module};
use candle_nn::{VarBuilder, VarMap, RmsNorm};
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
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub use_flash_attn: bool,
    pub rope_scaling: Option<serde_json::Value>,
}

fn linear(size_in: usize, size_out: usize, vb: VarBuilder) -> Result<AdapterLayer> {
    let w = vb.get((size_out, size_in), "weight")?;
    Ok(AdapterLayer::Linear(candle_nn::Linear::new(w, None)))
}

struct RotaryEmbedding { cos: Tensor, sin: Tensor }

impl RotaryEmbedding {
    fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let dim = cfg.hidden_size / cfg.num_attention_heads;
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

/// Phi-4 uses fused QKV projection (qkv_proj) same as Phi-3
pub struct Phi4Attention {
    pub qkv_proj: AdapterLayer,
    pub o_proj: AdapterLayer,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope: RotaryEmbedding,
}

impl Phi4Attention {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let hd = cfg.hidden_size / cfg.num_attention_heads;
        let q_dim = cfg.num_attention_heads * hd;
        let kv_dim = cfg.num_key_value_heads * hd;
        // Try fused qkv_proj first, fall back to separate q_proj/k_proj/v_proj
        let qkv_proj = if vb.contains_tensor("qkv_proj.weight") {
            linear(cfg.hidden_size, q_dim + 2 * kv_dim, vb.pp("qkv_proj"))?
        } else {
            return Err(candle_core::Error::Msg("Phi4 requires qkv_proj".to_string()));
        };
        Ok(Self {
            qkv_proj,
            o_proj: linear(q_dim, cfg.hidden_size, vb.pp("o_proj"))?,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: hd,
            rope: RotaryEmbedding::new(vb.dtype(), cfg, vb.device())?,
        })
    }

    fn forward(&self, x: &Tensor, pos: usize, cache: &mut Cache, layer_idx: usize) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let qkv = self.qkv_proj.forward(x)?;
        let q_dim = self.num_heads * self.head_dim;
        let kv_dim = self.num_kv_heads * self.head_dim;
        let q = qkv.narrow(candle_core::D::Minus1, 0, q_dim)?;
        let k = qkv.narrow(candle_core::D::Minus1, q_dim, kv_dim)?;
        let v = qkv.narrow(candle_core::D::Minus1, q_dim + kv_dim, kv_dim)?;
        let q = q.reshape((b, s, self.num_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let k = k.reshape((b, s, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let v = v.reshape((b, s, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
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
        let k = if n_rep > 1 { let (bk,nk,sk,dk)=k.dims4()?; k.unsqueeze(2)?.expand((bk,nk,n_rep,sk,dk))?.reshape((bk,nk*n_rep,sk,dk))? } else { k };
        let v = if n_rep > 1 { let (bv,nv,sv,dv)=v.dims4()?; v.unsqueeze(2)?.expand((bv,nv,n_rep,sv,dv))?.reshape((bv,nv*n_rep,sv,dv))? } else { v };
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn = candle_nn::ops::softmax(&(q.matmul(&k.t()?)? * scale)?, candle_core::D::Minus1)?;
        attn.matmul(&v)?.transpose(1, 2)?.reshape((b, s, self.num_heads * self.head_dim)).and_then(|y| self.o_proj.forward(&y))
    }
}

/// Fused gate_up_proj MLP (same approach as Phi-3)
pub struct Phi4MLP {
    pub gate_up_proj: AdapterLayer,
    pub down_proj: AdapterLayer,
    intermediate_size: usize,
}

impl Phi4MLP {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        Ok(Self {
            gate_up_proj: linear(cfg.hidden_size, 2 * cfg.intermediate_size, vb.pp("gate_up_proj"))?,
            down_proj: linear(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?,
            intermediate_size: cfg.intermediate_size,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate_up = self.gate_up_proj.forward(x)?;
        let gate = gate_up.narrow(candle_core::D::Minus1, 0, self.intermediate_size)?;
        let up = gate_up.narrow(candle_core::D::Minus1, self.intermediate_size, self.intermediate_size)?;
        let swiglu = unsloth_rs::kernels::swiglu_cubecl(&gate, &up)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        self.down_proj.forward(&swiglu)
    }
}

pub struct Phi4Block {
    pub attn: Phi4Attention,
    pub mlp: Phi4MLP,
    pub input_layernorm: RmsNorm,
    pub post_attention_layernorm: RmsNorm,
}

impl Phi4Block {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        Ok(Self {
            input_layernorm: RmsNorm::new(vb.pp("input_layernorm").get(cfg.hidden_size, "weight")?, cfg.rms_norm_eps),
            attn: Phi4Attention::load(vb.pp("self_attn"), cfg)?,
            post_attention_layernorm: RmsNorm::new(vb.pp("post_attention_layernorm").get(cfg.hidden_size, "weight")?, cfg.rms_norm_eps),
            mlp: Phi4MLP::load(vb.pp("mlp"), cfg)?,
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

pub struct Phi4 {
    pub embed_tokens: candle_nn::Embedding,
    pub layers: Vec<Phi4Block>,
    pub norm: RmsNorm,
    pub lm_head: AdapterLayer,
}

impl Phi4 {
    pub fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let embed_weight = vb.pp("model.embed_tokens").get((cfg.vocab_size, cfg.hidden_size), "weight")?;
        let embed_tokens = candle_nn::Embedding::new(embed_weight.clone(), cfg.hidden_size);
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| Phi4Block::load(vb.pp(&format!("model.layers.{}", i)), cfg))
            .collect::<Result<Vec<_>>>()?;
        let norm = RmsNorm::new(vb.pp("model.norm").get(cfg.hidden_size, "weight")?, cfg.rms_norm_eps);
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
            lora!(layer.attn.qkv_proj, "self_attn.qkv_proj");
            lora!(layer.attn.o_proj, "self_attn.o_proj");
            lora!(layer.mlp.gate_up_proj, "mlp.gate_up_proj");
            lora!(layer.mlp.down_proj, "mlp.down_proj");
        }
        Ok(())
    }
}

pub struct Phi4Model {
    pub model: Phi4,
    pub config: Config,
    pub device: Device,
    pub dtype: DType,
    pub cache: Cache,
    pub varmap: VarMap,
}

impl Phi4Model {
    pub fn new(model: Phi4, config: Config, device: Device, dtype: DType, varmap: VarMap) -> Self {
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
