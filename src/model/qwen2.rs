use candle_core::{DType, Device, Result, Tensor, Module, Var};
use candle_nn::{Activation, VarBuilder, RmsNorm, VarMap};
use crate::model::llama::{Cache}; 
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
    pub tie_word_embeddings: bool,
    pub use_flash_attn: bool,
    pub rope_scaling: Option<(String, f64)>,
}

pub fn linear(size1: usize, size2: usize, vb: VarBuilder) -> Result<AdapterLayer> {
    let weight = vb.get((size2, size1), "weight")?;
    // Qwen2 might have bias. Try to load it.
    let bias = vb.get((size2,), "bias").ok();
    let l = candle_nn::Linear::new(weight, bias);
    Ok(AdapterLayer::Linear(l))
}

struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let dim = cfg.hidden_size / cfg.num_attention_heads;
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim / 2)
            .map(|i| 1.0 / cfg.rope_theta.powf(2.0 * i as f32 / dim as f32))
            .collect();
        let inv_freq_len = inv_freq.len();
        
        let mut inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        
        if let Some((typ, factor)) = &cfg.rope_scaling {
             if typ == "linear" {
                 inv_freq = (inv_freq / *factor)?;
             } else if typ == "dynamic" {
                 let scale = 1.0 / factor;
                 let inv_freq_cpu = (0..dim / 2)
                    .map(|i| 1.0 / (cfg.rope_theta * scale as f32).powf(2.0 * i as f32 / dim as f32))
                    .collect::<Vec<_>>();
                 inv_freq = Tensor::from_vec(inv_freq_cpu, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
             }
        }

        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let cos = freqs.cos()?;
        let sin = freqs.sin()?;
        Ok(Self { cos, sin })
    }

    fn forward(&self, x: &Tensor, pos: usize, seq_len: usize) -> Result<Tensor> {
        let (_b, _s, _h, _d) = x.dims4()?;
        let cos = self.cos.narrow(0, pos, seq_len)?;
        let sin = self.sin.narrow(0, pos, seq_len)?;
        candle_nn::rotary_emb::rope(x, &cos, &sin)
    }
}

pub struct Qwen2Attention {
    pub q_proj: AdapterLayer,
    pub k_proj: AdapterLayer,
    pub v_proj: AdapterLayer,
    pub o_proj: AdapterLayer,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    rotary_emb: RotaryEmbedding,
    use_flash_attn: bool,
}

impl Qwen2Attention {
    pub fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let size_in = cfg.hidden_size;
        let size_q = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_attention_heads;
        let size_kv = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;
        
        let q_proj = linear(size_in, size_q, vb.pp("q_proj"))?;
        let k_proj = linear(size_in, size_kv, vb.pp("k_proj"))?;
        let v_proj = linear(size_in, size_kv, vb.pp("v_proj"))?;
        let o_proj = linear(size_q, size_in, vb.pp("o_proj"))?;
        
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let rotary_emb = RotaryEmbedding::new(vb.dtype(), cfg, vb.device())?;
        
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim,
            rotary_emb,
            use_flash_attn: cfg.use_flash_attn,
        })
    }

    pub fn forward(&self, x: &Tensor, pos: usize, cache: &mut Cache, layer_idx: usize) -> Result<Tensor> {
        let (b_sz, seq_len, hidden_size) = x.dims3()?;
        
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q.reshape((b_sz, seq_len, self.num_attention_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let k = k.reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let v = v.reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;

        let q = self.rotary_emb.forward(&q, pos, seq_len)?;
        let k = self.rotary_emb.forward(&k, pos, seq_len)?;

        let (k, v) = if cache.use_kv_cache {
            let (k, v) = match &cache.kvs[layer_idx] {
                Some((prev_k, prev_v)) => {
                    let k = Tensor::cat(&[prev_k, &k], 2)?;
                    let v = Tensor::cat(&[prev_v, &v], 2)?;
                    (k, v)
                }
                None => (k, v),
            };
            cache.kvs[layer_idx] = Some::<(Tensor, Tensor)>((k.clone(), v.clone()));
            (k, v)
        } else {
            (k, v)
        };

        let k = self.repeat_kv(k)?;
        let v = self.repeat_kv(v)?;

        let y = if self.use_flash_attn {
             #[cfg(feature = "flash-attn")]
             {
                 let q = q.transpose(1, 2)?.contiguous()?; 
                 let k = k.transpose(1, 2)?.contiguous()?;
                 let v = v.transpose(1, 2)?.contiguous()?;
                 let softmax_scale = 1.0 / (self.head_dim as f64).sqrt();
                 candle_flash_attn::flash_attn(&q, &k, &v, softmax_scale as f32, true)?
             }
             #[cfg(not(feature = "flash-attn"))]
             {
                 Self::naive_attn(&q, &k, &v, self.head_dim)?
             }
        } else {
             Self::naive_attn(&q, &k, &v, self.head_dim)?
        };
        
        let y = y.reshape((b_sz, seq_len, hidden_size))?;
        let y = self.o_proj.forward(&y)?;
        Ok(y)
    }

    fn naive_attn(q: &Tensor, k: &Tensor, v: &Tensor, head_dim: usize) -> Result<Tensor> {
        let scale = 1.0 / (head_dim as f64).sqrt();
        let attn_weights = (q.matmul(&k.t()?)? * scale)?;
        let attn_weights = candle_nn::ops::softmax(&attn_weights, candle_core::D::Minus1)?;
        let y = attn_weights.matmul(v)?; 
        y.transpose(1, 2) 
    }
    
    fn repeat_kv(&self, x: Tensor) -> Result<Tensor> {
        let n_rep = self.num_attention_heads / self.num_key_value_heads;
        if n_rep == 1 {
            return Ok(x);
        }
        let (b, n_kv, s, d) = x.dims4()?;
        x.unsqueeze(2)?
         .expand((b, n_kv, n_rep, s, d))?
         .reshape((b, n_kv * n_rep, s, d))
    }
}

pub struct Qwen2MLP {
    pub gate_proj: AdapterLayer,
    pub up_proj: AdapterLayer,
    pub down_proj: AdapterLayer,
    pub act_fn: Activation,
}

impl Qwen2MLP {
    pub fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let intermediate_size = cfg.intermediate_size;
        
        let gate_proj = linear(hidden_size, intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = linear(hidden_size, intermediate_size, vb.pp("up_proj"))?;
        let down_proj = linear(intermediate_size, hidden_size, vb.pp("down_proj"))?;
        
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: Activation::Silu,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let lhs = self.gate_proj.forward(x)?.apply(&self.act_fn)?;
        let rhs = self.up_proj.forward(x)?;
        self.down_proj.forward(&(lhs * rhs)?)
    }
}

pub struct Qwen2Block {
    attn: Qwen2Attention,
    mlp: Qwen2MLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl Qwen2Block {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let input_layernorm = RmsNorm::new(vb.pp("input_layernorm").get(cfg.hidden_size, "weight")?, cfg.rms_norm_eps);
        let attn = Qwen2Attention::load(vb.pp("self_attn"), cfg)?;
        let post_attention_layernorm = RmsNorm::new(vb.pp("post_attention_layernorm").get(cfg.hidden_size, "weight")?, cfg.rms_norm_eps);
        let mlp = Qwen2MLP::load(vb.pp("mlp"), cfg)?;
        
        Ok(Self {
            attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(&self, x: &Tensor, pos: usize, cache: &mut Cache, layer_idx: usize) -> Result<Tensor> {
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let x = self.attn.forward(&x, pos, cache, layer_idx)?;
        let x = (x + residual)?;
        
        let residual = &x;
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        let x = (x + residual)?;
        Ok(x)
    }
}

pub struct Qwen2 {
    pub embed_tokens: candle_nn::Embedding,
    pub layers: Vec<Qwen2Block>,
    pub norm: RmsNorm,
    pub lm_head: AdapterLayer,
}

impl Qwen2 {
    pub fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let embed_tokens = candle_nn::Embedding::new(vb.pp("model.embed_tokens").get((cfg.vocab_size, cfg.hidden_size), "weight")?, cfg.hidden_size);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(Qwen2Block::load(vb.pp(&format!("model.layers.{}", i)), cfg)?);
        }
        let norm = RmsNorm::new(vb.pp("model.norm").get(cfg.hidden_size, "weight")?, cfg.rms_norm_eps);
        let lm_head = linear(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?;
        
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head
        })
    }
    
    pub fn forward(&mut self, input_ids: &Tensor, pos: usize, cache: &mut Cache) -> Result<Tensor> {
        let (_b_sz, seq_len) = input_ids.dims2()?;
        let mut x = self.embed_tokens.forward(input_ids)?;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            x = layer.forward(&x, pos, cache, i)?;
        }
        let x = self.norm.forward(&x)?;
        let logits = self.lm_head.forward(&x)?;
        Ok(logits)
    }

    pub fn apply_lora(&mut self, target_modules: Vec<String>, rank: usize, alpha: f64, _dropout: f64, use_dora: bool, varmap: &mut VarMap) -> Result<()> {
        let scaling = alpha / rank as f64;
        let device = self.embed_tokens.embeddings().device().clone();
        let dtype = self.embed_tokens.embeddings().dtype();
        
        for (i, layer) in self.layers.iter_mut().enumerate() {
             use crate::model::inject_lora;
             if target_modules.contains(&"q_proj".to_string()) {
                inject_lora(&mut layer.attn.q_proj, rank, scaling, varmap, &device, dtype, format!("model.layers.{}.self_attn.q_proj", i), use_dora)?;
            }
            if target_modules.contains(&"k_proj".to_string()) {
                inject_lora(&mut layer.attn.k_proj, rank, scaling, varmap, &device, dtype, format!("model.layers.{}.self_attn.k_proj", i), use_dora)?;
            }
            if target_modules.contains(&"v_proj".to_string()) {
                inject_lora(&mut layer.attn.v_proj, rank, scaling, varmap, &device, dtype, format!("model.layers.{}.self_attn.v_proj", i), use_dora)?;
            }
            if target_modules.contains(&"o_proj".to_string()) {
                inject_lora(&mut layer.attn.o_proj, rank, scaling, varmap, &device, dtype, format!("model.layers.{}.self_attn.o_proj", i), use_dora)?;
            }
            
            if target_modules.contains(&"gate_proj".to_string()) {
                inject_lora(&mut layer.mlp.gate_proj, rank, scaling, varmap, &device, dtype, format!("model.layers.{}.mlp.gate_proj", i), use_dora)?;
            }
            if target_modules.contains(&"up_proj".to_string()) {
                inject_lora(&mut layer.mlp.up_proj, rank, scaling, varmap, &device, dtype, format!("model.layers.{}.mlp.up_proj", i), use_dora)?;
            }
            if target_modules.contains(&"down_proj".to_string()) {
                inject_lora(&mut layer.mlp.down_proj, rank, scaling, varmap, &device, dtype, format!("model.layers.{}.mlp.down_proj", i), use_dora)?;
            }
        }
        Ok(())
    }
}

pub struct Qwen2Model {
    pub model: Qwen2, 
    pub config: Config,
    pub device: Device,
    pub dtype: DType,
    pub cache: Cache,
    pub varmap: VarMap,
}

impl Qwen2Model {
    pub fn new(model: Qwen2, config: Config, device: Device, dtype: DType, varmap: VarMap) -> Self {
        let cache = Cache::new(true, config.num_hidden_layers);
        Self {
            model,
            config,
            device,
            dtype,
            cache,
            varmap,
        }
    }
    
    pub fn forward(&mut self, input_ids: &Tensor, pos: usize) -> Result<Tensor> {
        self.model.forward(input_ids, pos, &mut self.cache)
    }
    
    pub fn clear_cache(&mut self) {
         self.cache = Cache::new(true, self.config.num_hidden_layers);
    }

    pub fn apply_lora(&mut self, target_modules: Vec<String>, rank: usize, alpha: f64, dropout: f64, use_dora: bool) -> Result<()> {
        self.model.apply_lora(target_modules, rank, alpha, dropout, use_dora, &mut self.varmap)
    }
}
