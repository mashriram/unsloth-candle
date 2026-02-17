use candle_core::{DType, Device, Result, Tensor, Module};
use candle_nn::{Embedding, VarBuilder, Activation, VarMap};


// use candle_transformers::models::llama::Config; // Use local config

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
    pub use_gradient_checkpointing: bool,
    pub load_in_4bit: bool,
}

use crate::model::layers::{AdapterLayer, UnslothRmsNorm};

// Helper to create Linear layer (no bias usually for Llama)
fn linear(size1: usize, size2: usize, vb: VarBuilder, cfg: &Config) -> Result<AdapterLayer> {
    let weight = vb.get((size2, size1), "weight")?;
    if cfg.load_in_4bit {
        let l = crate::model::linear4bit::Linear4bit::from_tensor(&weight, 64)?; // Block size 64 hardcoded for now
        Ok(AdapterLayer::Linear4bit(l))
    } else {
        let l = candle_nn::Linear::new(weight, None);
        Ok(AdapterLayer::Linear(l))
    }
}

#[derive(Clone)]
pub struct Cache {
    pub kvs: Vec<Option<(Tensor, Tensor)>>,
    pub use_kv_cache: bool,
}

impl Cache {
    pub fn new(use_kv_cache: bool, num_layers: usize) -> Self {
        Self {
            kvs: vec![None; num_layers],
            use_kv_cache,
        }
    }
}

// ... RotaryEmbedding ...
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
        unsloth_rs::kernels::rope_cubecl(x, &cos, &sin)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))
    }
}

pub struct CausalSelfAttention {
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

impl CausalSelfAttention {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let size_in = cfg.hidden_size;
        let size_q = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_attention_heads;
        let size_kv = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;
        
        let q_proj = linear(size_in, size_q, vb.pp("q_proj"), cfg)?;
        let k_proj = linear(size_in, size_kv, vb.pp("k_proj"), cfg)?;
        let v_proj = linear(size_in, size_kv, vb.pp("v_proj"), cfg)?;
        let o_proj = linear(size_q, size_in, vb.pp("o_proj"), cfg)?;
        
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

    fn forward(&self, x: &Tensor, pos: usize, cache: &mut Cache, layer_idx: usize) -> Result<Tensor> {
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
                 let q = q.transpose(1, 2)?.contiguous()?; // [b, s, h, d]
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

#[derive(Clone)]
pub struct Mlp {
    pub gate_proj: AdapterLayer,
    pub up_proj: AdapterLayer,
    pub down_proj: AdapterLayer,
}

impl Mlp {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let intermediate_size = cfg.intermediate_size;
        
        let gate_proj = linear(hidden_size, intermediate_size, vb.pp("gate_proj"), cfg)?;
        let up_proj = linear(hidden_size, intermediate_size, vb.pp("up_proj"), cfg)?;
        let down_proj = linear(intermediate_size, hidden_size, vb.pp("down_proj"), cfg)?;
        
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
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

pub struct Block {
    pub rms_1: UnslothRmsNorm,
    pub attn: CausalSelfAttention,
    pub rms_2: UnslothRmsNorm,
    pub mlp: Mlp,
}


impl Block {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let rms_1 = UnslothRmsNorm::new(vb.pp("input_layernorm").get(cfg.hidden_size, "weight")?, cfg.rms_norm_eps);
        let attn = CausalSelfAttention::load(vb.pp("self_attn"), cfg)?;
        let rms_2 = UnslothRmsNorm::new(vb.pp("post_attention_layernorm").get(cfg.hidden_size, "weight")?, cfg.rms_norm_eps);
        let mlp = Mlp::load(vb.pp("mlp"), cfg)?;
        Ok(Self { rms_1, attn, rms_2, mlp })
    }

    pub fn forward(&self, x: &Tensor, pos: usize, cache: &mut Cache, layer_idx: usize) -> Result<Tensor> {
        let residual = x;
        let x = self.rms_1.forward(x)?;
        let x = self.attn.forward(&x, pos, cache, layer_idx)?;
        let x = (x + residual)?;
        
        let residual = &x;
        let x = self.rms_2.forward(&x)?;
        
        // Checkpointing for MLP
        // We capture the necessary parts for MLP forward in a closure
        // Note: MLP forward only depends on `x`.
        // However, `checkpoint` function takes `x` and a closure `f(x)`.
        // The closure must be Send + Sync + 'static (ish).
        // `self.mlp` is a reference, so we can't move it easily unless we clone or use Arc.
        // `Mlp` contains `AdapterLayer`s which are likely clone-able (Linear is Arc-ed?).
        // Let's check AdapterLayer cloneability. It should be cheap.
        
        // TODO: For now, we apply checkpointing only if a config flag is set? 
        // Or we expose a method `forward_checkpointed`?
        // User asked to "Implement Gradient Checkpointing".
        // Let's add a `use_gradient_checkpointing` flag to Config or similar?
        // For now, let's just implement the logic in a way that can be toggled or used.
        // Since we don't have a flag yet, I'll add a comment and maybe a `forward_with_checkpointing` method?
        // Or just replace the MLP call for now to test it.
        // Wait, checkpointing usually wraps the whole block or large parts.
        // Wrapping just MLP is okay. Wrapping Attention + MLP is better.
        // But Attention has side effects (KV Cache).
        // Gradient Checkpointing with KV Cache is tricky because the cache update happens in forward.
        // If we re-run forward, we might double-update cache?
        // Usually, for checkpointing, we don't checkpoint the Attention part with KV cache update, 
        // OR we ensure the cache update is idempotent / ignored in re-computation.
        // But `candle` KV cache is passed as `&mut Cache`.
        // If we re-run, we seek `&mut Cache`.
        // This is unsafe/hard with standard closures.
        
        // Strategy: Only checkpoint the MLP part for now. It's stateless.
        
        // let mlp = self.mlp.clone(); // Assuming Mlp is Clone (AdapterLayer is Clone?)
        // let x_out = crate::core::checkpoint::checkpoint(Arc::new(move |t| mlp.forward(t)), &x)?;
        
        let x = self.mlp.forward(&x)?;
        let x = (x + residual)?;
        Ok(x)
    }

    pub fn forward_checkpointed(&self, x: &Tensor, pos: usize, cache: &mut Cache, layer_idx: usize) -> Result<Tensor> {
        let residual = x;
        let x = self.rms_1.forward(x)?;
        let x = self.attn.forward(&x, pos, cache, layer_idx)?;
        let x = (x + residual)?;
        
        let residual = &x;
        let x = self.rms_2.forward(&x)?;
        
        // Gradient Checkpointing for MLP
        // We move a clone of MLP (cheap, just Arc tensors) into the closure.
        let mlp = self.mlp.clone();
        let x = crate::core::checkpoint::checkpoint(
            std::sync::Arc::new(move |t| mlp.forward(t)), 
            &x
        )?;
        
        let x = (x + residual)?;
        Ok(x)
    }
}

pub struct Llama {
    pub embed_tokens: Embedding,
    pub layers: Vec<Block>,
    pub norm: UnslothRmsNorm,
    pub lm_head: AdapterLayer,
    pub device: Device,
    pub dtype: DType,
    pub use_gradient_checkpointing: bool,
}

impl Llama {
    pub fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let embed_tokens = Embedding::new(vb.pp("model.embed_tokens").get((cfg.vocab_size, cfg.hidden_size), "weight")?, cfg.hidden_size);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(Block::load(vb.pp(&format!("model.layers.{}", i)), cfg)?);
        }
        let norm = UnslothRmsNorm::new(vb.pp("model.norm").get(cfg.hidden_size, "weight")?, cfg.rms_norm_eps);
        
        let lm_head = if cfg.tie_word_embeddings {
             let weight = embed_tokens.embeddings().clone();
             let l = candle_nn::Linear::new(weight, None);
             AdapterLayer::Linear(l)
        } else {
             linear(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"), cfg)?
        };
        
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            use_gradient_checkpointing: cfg.use_gradient_checkpointing,
        })
    }

    pub fn forward(&mut self, input_ids: &Tensor, pos: usize, cache: &mut Cache) -> Result<Tensor> {
        let (_b_sz, _seq_len) = input_ids.dims2()?;
        let mut x = self.embed_tokens.forward(input_ids)?;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            if self.use_gradient_checkpointing {
                x = layer.forward_checkpointed(&x, pos, cache, i)?;
            } else {
                x = layer.forward(&x, pos, cache, i)?;
            }
        }
        let x = self.norm.forward(&x)?;
        // Return full sequence logits for training
        // let x = x.narrow(1, seq_len - 1, 1)?; 
        let logits = self.lm_head.forward(&x)?;
        Ok(logits)
    }

    pub fn apply_lora(&mut self, target_modules: Vec<String>, rank: usize, alpha: f64, _dropout: f64, use_dora: bool, varmap: &mut VarMap) -> Result<()> {
        let scaling = alpha / rank as f64;
        let device = self.device.clone();
        let dtype = self.dtype;
        
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
