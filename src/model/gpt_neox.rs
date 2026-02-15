use candle_core::{DType, Device, Result, Tensor, Module, IndexOp};
use candle_nn::{Activation, VarBuilder, VarMap, LayerNorm};
use crate::model::layers::AdapterLayer;

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub layer_norm_eps: f64,
    pub rotary_pct: f64,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub use_parallel_residual: bool,
}

// Helper to create Linear layer with bias 
fn linear(size1: usize, size2: usize, vb: VarBuilder) -> Result<AdapterLayer> {
    let weight = vb.get((size2, size1), "weight")?;
    let bias = vb.get((size2,), "bias")?;
    let l = candle_nn::Linear::new(weight, Some(bias));
    Ok(AdapterLayer::Linear(l))
}

struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
    rotary_dim: usize,
}

impl RotaryEmbedding {
    fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let rotary_dim = (head_dim as f64 * cfg.rotary_pct) as usize;
        let max_seq_len = cfg.max_position_embeddings;
        
        let inv_freq: Vec<_> = (0..rotary_dim / 2)
            .map(|i| 1.0 / cfg.rope_theta.powf(2.0 * i as f32 / rotary_dim as f32))
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, (1, rotary_dim / 2), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let cos = freqs.cos()?;
        let sin = freqs.sin()?;
        Ok(Self { cos, sin, rotary_dim })
    }

    fn forward(&self, x: &Tensor, pos: usize, seq_len: usize) -> Result<Tensor> {
        // x: [b, seq, heads, head_dim]
        let (_b, _seq, _h, dim) = x.dims4()?;
        
        let x_rot = x.narrow(candle_core::D::Minus1, 0, self.rotary_dim)?;
        let x_pass = x.narrow(candle_core::D::Minus1, self.rotary_dim, dim - self.rotary_dim)?;
        
        let cos = self.cos.narrow(0, pos, seq_len)?;
        let sin = self.sin.narrow(0, pos, seq_len)?;
        
        // Rope expects [b, seq, heads, dim]
        let x_rotated = candle_nn::rotary_emb::rope(&x_rot, &cos, &sin)?;
        
        Tensor::cat(&[&x_rotated, &x_pass], candle_core::D::Minus1)
    }
}

pub struct GPTNeoXAttention {
    pub query_key_value: AdapterLayer,
    pub dense: AdapterLayer,
    pub num_attention_heads: usize,
    pub head_dim: usize,
    rotary_emb: RotaryEmbedding,
}

impl GPTNeoXAttention {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let head_dim = hidden / cfg.num_attention_heads;
        let rotary_emb = RotaryEmbedding::new(vb.dtype(), cfg, vb.device())?;
        
        let query_key_value = linear(hidden, 3 * hidden, vb.pp("query_key_value"))?;
        let dense = linear(hidden, hidden, vb.pp("dense"))?;
        
        Ok(Self {
            query_key_value,
            dense,
            num_attention_heads: cfg.num_attention_heads,
            head_dim,
            rotary_emb,
        })
    }

    fn forward(&self, x: &Tensor, pos: usize) -> Result<Tensor> {
        let (b, seq, _hidden) = x.dims3()?;
        let qkv = self.query_key_value.forward(x)?;
        
        let qkv = qkv.reshape((b, seq, 3, self.num_attention_heads, self.head_dim))?
                     .transpose(1, 2)?; // [b, 3, seq, heads, head_dim]
        
        let q = qkv.i(0)?;
        let k = qkv.i(1)?;
        let v = qkv.i(2)?;
        
        // Q: [b, seq, heads, dim] for RoPE
        let q = self.rotary_emb.forward(&q, pos, seq)?;
        let k = self.rotary_emb.forward(&k, pos, seq)?;
        
        // Flash Attn or Naive
        // Transpose to [b, heads, seq, dim] for attn
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;
        
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let att = (q.matmul(&k.t()?)? * scale)?;
        let att = candle_nn::ops::softmax(&att, candle_core::D::Minus1)?;
        let y = att.matmul(&v)?;
        
        let y = y.transpose(1, 2)?.reshape((b, seq, self.num_attention_heads * self.head_dim))?;
        self.dense.forward(&y)
    }
}

pub struct GPTNeoXMLP {
    pub dense_h_to_4h: AdapterLayer,
    pub dense_4h_to_h: AdapterLayer,
    pub act: Activation,
}

impl GPTNeoXMLP {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let dense_h_to_4h = linear(hidden, inter, vb.pp("dense_h_to_4h"))?;
        let dense_4h_to_h = linear(inter, hidden, vb.pp("dense_4h_to_h"))?;
        Ok(Self {
            dense_h_to_4h,
            dense_4h_to_h,
            act: Activation::Gelu,
        })
    }
    
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.dense_h_to_4h.forward(x)?;
        let x = x.apply(&self.act)?;
        self.dense_4h_to_h.forward(&x)
    }
}

pub struct GPTNeoXLayer {
    pub input_layernorm: LayerNorm,
    pub post_attention_layernorm: LayerNorm,
    pub attention: GPTNeoXAttention,
    pub mlp: GPTNeoXMLP,
    pub use_parallel_residual: bool,
}

impl GPTNeoXLayer {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let input_layernorm = candle_nn::layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = candle_nn::layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("post_attention_layernorm"))?;
        let attention = GPTNeoXAttention::load(vb.pp("attention"), cfg)?;
        let mlp = GPTNeoXMLP::load(vb.pp("mlp"), cfg)?;
        
        Ok(Self {
            input_layernorm,
            post_attention_layernorm,
            attention,
            mlp,
            use_parallel_residual: cfg.use_parallel_residual
        })
    }
    
    fn forward(&self, x: &Tensor, pos: usize) -> Result<Tensor> {
        if self.use_parallel_residual {
             let ln1 = self.input_layernorm.forward(x)?;
             let ln2 = self.post_attention_layernorm.forward(x)?;
             let attn_out = self.attention.forward(&ln1, pos)?;
             let mlp_out = self.mlp.forward(&ln2)?;
             x + attn_out + mlp_out
        } else {
             let residual = x;
             let ln1 = self.input_layernorm.forward(x)?;
             let attn_out = self.attention.forward(&ln1, pos)?;
             let x = (residual + attn_out)?;
             
             let residual = &x;
             let ln2 = self.post_attention_layernorm.forward(&x)?;
             let mlp_out = self.mlp.forward(&ln2)?;
             x + mlp_out
        }
    }
}

pub struct GPTNeoX {
    pub embed_in: candle_nn::Embedding,
    pub layers: Vec<GPTNeoXLayer>,
    pub final_layer_norm: LayerNorm,
    pub embed_out: AdapterLayer, // Shared usually
}

impl GPTNeoX {
    pub fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let embed_in = candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_in"))?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(GPTNeoXLayer::load(vb.pp(&format!("layers.{}", i)), cfg)?);
        }
        let final_layer_norm = candle_nn::layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("final_layer_norm"))?;
        let embed_out = linear(cfg.hidden_size, cfg.vocab_size, vb.pp("embed_out"))?;
        
        Ok(Self {
            embed_in,
            layers,
            final_layer_norm,
            embed_out
        })
    }
    
    pub fn forward(&mut self, input_ids: &Tensor, pos: usize) -> Result<Tensor> {
        let (_b, _seq) = input_ids.dims2()?;
        let mut x = self.embed_in.forward(input_ids)?;
        for layer in &self.layers {
            x = layer.forward(&x, pos)?;
        }
        let x = self.final_layer_norm.forward(&x)?;
        self.embed_out.forward(&x)
    }
    
     pub fn apply_lora(&mut self, target_modules: Vec<String>, rank: usize, alpha: f64, _dropout: f64, use_dora: bool, varmap: &mut VarMap) -> Result<()> {
         let scaling = alpha / rank as f64;
         let device = self.embed_in.embeddings().device().clone();
         let dtype = self.embed_in.embeddings().dtype();
         
         for (i, layer) in self.layers.iter_mut().enumerate() {
              use crate::model::inject_lora;
              if target_modules.contains(&"query_key_value".to_string()) {
                 inject_lora(&mut layer.attention.query_key_value, rank, scaling, varmap, &device, dtype, format!("layers.{}.attention.query_key_value", i), use_dora)?;
              }
              if target_modules.contains(&"dense".to_string()) {
                 inject_lora(&mut layer.attention.dense, rank, scaling, varmap, &device, dtype, format!("layers.{}.attention.dense", i), use_dora)?;
              }
              if target_modules.contains(&"dense_h_to_4h".to_string()) {
                 inject_lora(&mut layer.mlp.dense_h_to_4h, rank, scaling, varmap, &device, dtype, format!("layers.{}.mlp.dense_h_to_4h", i), use_dora)?;
              }
              if target_modules.contains(&"dense_4h_to_h".to_string()) {
                 inject_lora(&mut layer.mlp.dense_4h_to_h, rank, scaling, varmap, &device, dtype, format!("layers.{}.mlp.dense_4h_to_h", i), use_dora)?;
              }
         }
         Ok(())
     }
}

pub struct GPTNeoXModel {
    pub model: GPTNeoX,
    pub config: Config,
    pub device: Device,
    pub dtype: DType,
    pub varmap: VarMap,
}

impl GPTNeoXModel {
    pub fn new(model: GPTNeoX, config: Config, device: Device, dtype: DType, varmap: VarMap) -> Self {
        Self {
            model,
            config,
            device,
            dtype,
            varmap,
        }
    }
    
    pub fn forward(&mut self, input_ids: &Tensor, pos: usize) -> Result<Tensor> {
        self.model.forward(input_ids, pos)
    }
    
    pub fn apply_lora(&mut self, target_modules: Vec<String>, rank: usize, alpha: f64, dropout: f64, use_dora: bool) -> Result<()> {
        self.model.apply_lora(target_modules, rank, alpha, dropout, use_dora, &mut self.varmap)
    }
    
    pub fn clear_cache(&mut self) {}
}
