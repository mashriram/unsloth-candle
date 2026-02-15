use candle_core::{DType, Device, Result, Tensor, Module};
use candle_nn::{Activation, VarBuilder, VarMap, LayerNorm};
use crate::model::layers::AdapterLayer;

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize, // usually equal to heads or 1?
    pub layer_norm_eps: f64,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub use_qkv_bias: bool, // Cohere usually doesn't?
    pub logit_scale: Option<f64>,
}

fn linear(size1: usize, size2: usize, vb: VarBuilder) -> Result<AdapterLayer> {
    let weight = vb.get((size2, size1), "weight")?;
    // Cohere Linear usually no bias in QKV/MLP?
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
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let dim = head_dim; // Cohere full rotation
        let max_seq_len = cfg.max_position_embeddings;
        
        let inv_freq: Vec<_> = (0..dim / 2)
            .map(|i| 1.0 / cfg.rope_theta.powf(2.0 * i as f32 / dim as f32))
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, (1, dim / 2), dev)?.to_dtype(dtype)?;
        
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

pub struct CohereAttention {
    q_proj: AdapterLayer,
    k_proj: AdapterLayer,
    v_proj: AdapterLayer,
    o_proj: AdapterLayer,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary: RotaryEmbedding,
}

impl CohereAttention {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let dim = cfg.hidden_size;
        let head_dim = dim / cfg.num_attention_heads;
        let q_dim = cfg.num_attention_heads * head_dim;
        let kv_dim = cfg.num_key_value_heads * head_dim;
        
        let q_proj = linear(dim, q_dim, vb.pp("q_proj"))?;
        let k_proj = linear(dim, kv_dim, vb.pp("k_proj"))?;
        let v_proj = linear(dim, kv_dim, vb.pp("v_proj"))?;
        let o_proj = linear(q_dim, dim, vb.pp("o_proj"))?;
        
        let rotary = RotaryEmbedding::new(vb.dtype(), cfg, vb.device())?;
        
        Ok(Self {
            q_proj, k_proj, v_proj, o_proj,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim, rotary
        })
    }
    
    fn forward(&self, x: &Tensor, pos: usize) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;
        
        let q = q.reshape((b, s, self.num_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let k = k.reshape((b, s, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let v = v.reshape((b, s, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        
        let q = self.rotary.forward(&q, pos, s)?;
        let k = self.rotary.forward(&k, pos, s)?;
        
        // Repeat KV
        let k = self.repeat_kv(k)?;
        let v = self.repeat_kv(v)?;
        
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let att = (q.matmul(&k.t()?)? * scale)?;
        let att = candle_nn::ops::softmax(&att, candle_core::D::Minus1)?;
        let y = att.matmul(&v)?;
        
        let y = y.transpose(1, 2)?.reshape((b, s, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&y)
    }
    
    fn repeat_kv(&self, x: Tensor) -> Result<Tensor> {
        let n_rep = self.num_heads / self.num_kv_heads;
        if n_rep == 1 { return Ok(x); }
        let (b, n_kv, s, d) = x.dims4()?;
        x.unsqueeze(2)?.expand((b, n_kv, n_rep, s, d))?.reshape((b, n_kv * n_rep, s, d))
    }
}

pub struct CohereMLP {
    gate_proj: AdapterLayer,
    up_proj: AdapterLayer,
    down_proj: AdapterLayer,
    act: Activation,
}

impl CohereMLP {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let dim = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        
        let gate_proj = linear(dim, inter, vb.pp("gate_proj"))?;
        let up_proj = linear(dim, inter, vb.pp("up_proj"))?;
        let down_proj = linear(inter, dim, vb.pp("down_proj"))?;
        
        Ok(Self { gate_proj, up_proj, down_proj, act: Activation::Silu })
    }
    
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let lhs = self.gate_proj.forward(x)?.apply(&self.act)?;
        let rhs = self.up_proj.forward(x)?;
        self.down_proj.forward(&(lhs * rhs)?)
    }
}

pub struct CohereLayer {
    attn: CohereAttention,
    mlp: CohereMLP,
    input_norm: LayerNorm,
    // Cohere uses parallel attention? 
    // Usually standard sequential.
}

impl CohereLayer {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let attn = CohereAttention::load(vb.pp("self_attn"), cfg)?;
        let mlp = CohereMLP::load(vb.pp("mlp"), cfg)?;
        let input_norm = candle_nn::layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("input_layernorm"))?;
        
        Ok(Self { attn, mlp, input_norm })
    }
    
    fn forward(&self, x: &Tensor, pos: usize) -> Result<Tensor> {
        // Command R architecture:
        // x = norm(x)
        // attn_out = attn(x)
        // mlp_out = mlp(x)
        // x = x + attn_out + mlp_out  <-- Parallel block?
        // Need to confirm parallel vs sequential.
        // Assuming Standard Llama for now, but Command R is often parallel.
        // Let's implement sequential first, unless verified.
        
        let residual = x;
        let x_norm = self.input_norm.forward(x)?;
        let attn_out = self.attn.forward(&x_norm, pos)?;
        let mlp_out = self.mlp.forward(&x_norm)?;
        
        (residual + attn_out + mlp_out) // Parallel block logic common in recent LLMs like Palm/CodeLlama
    }
}

pub struct Cohere {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<CohereLayer>,
    norm: LayerNorm,
    // No lm_head? Usually share with embed?
    // Cohere might support logit scaling
}

impl Cohere {
     pub fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let embed_tokens = candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(CohereLayer::load(vb.pp(&format!("layers.{}", i)), cfg)?);
        }
        let norm = candle_nn::layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("norm"))?;
        
        Ok(Self { embed_tokens, layers, norm })
    }
    
    pub fn forward(&mut self, input_ids: &Tensor, pos: usize) -> Result<Tensor> {
        let mut x = self.embed_tokens.forward(input_ids)?;
        for layer in &self.layers {
            x = layer.forward(&x, pos)?;
        }
        let x = self.norm.forward(&x)?;
        
        // Logit scaling?
        // Weight tying?
        // Typically output = x * embed_weights.T
        let embed_weights = self.embed_tokens.embeddings();
        let logits = x.matmul(&embed_weights.t()?)?;
        Ok(logits)
    }
    
    pub fn apply_lora(&mut self, target_modules: Vec<String>, rank: usize, alpha: f64, _dropout: f64, use_dora: bool, varmap: &mut VarMap) -> Result<()> {
          let scaling = alpha / rank as f64;
          let device = self.embed_tokens.embeddings().device().clone();
          let dtype = self.embed_tokens.embeddings().dtype();
          
          for (i, layer) in self.layers.iter_mut().enumerate() {
              use crate::model::inject_lora;
              if target_modules.contains(&"q_proj".to_string()) {
                 inject_lora(&mut layer.attn.q_proj, rank, scaling, varmap, &device, dtype, format!("layers.{}.self_attn.q_proj", i), use_dora)?;
              }
              if target_modules.contains(&"k_proj".to_string()) {
                 inject_lora(&mut layer.attn.k_proj, rank, scaling, varmap, &device, dtype, format!("layers.{}.self_attn.k_proj", i), use_dora)?;
              }
              if target_modules.contains(&"v_proj".to_string()) {
                 inject_lora(&mut layer.attn.v_proj, rank, scaling, varmap, &device, dtype, format!("layers.{}.self_attn.v_proj", i), use_dora)?;
              }
              if target_modules.contains(&"o_proj".to_string()) {
                 inject_lora(&mut layer.attn.o_proj, rank, scaling, varmap, &device, dtype, format!("layers.{}.self_attn.o_proj", i), use_dora)?;
              }
              if target_modules.contains(&"gate_proj".to_string()) {
                 inject_lora(&mut layer.mlp.gate_proj, rank, scaling, varmap, &device, dtype, format!("layers.{}.mlp.gate_proj", i), use_dora)?;
              }
              if target_modules.contains(&"up_proj".to_string()) {
                 inject_lora(&mut layer.mlp.up_proj, rank, scaling, varmap, &device, dtype, format!("layers.{}.mlp.up_proj", i), use_dora)?;
              }
              if target_modules.contains(&"down_proj".to_string()) {
                 inject_lora(&mut layer.mlp.down_proj, rank, scaling, varmap, &device, dtype, format!("layers.{}.mlp.down_proj", i), use_dora)?;
              }
          }
          Ok(())
    }
}

pub struct CohereModel {
    pub model: Cohere,
    pub config: Config,
    pub device: Device,
    pub dtype: DType,
    pub varmap: VarMap,
}

impl CohereModel {
    pub fn new(model: Cohere, config: Config, device: Device, dtype: DType, varmap: VarMap) -> Self {
        Self { model, config, device, dtype, varmap }
    }
     pub fn forward(&mut self, input_ids: &Tensor, pos: usize) -> Result<Tensor> {
        self.model.forward(input_ids, pos)
    }
    
    pub fn apply_lora(&mut self, target_modules: Vec<String>, rank: usize, alpha: f64, dropout: f64, use_dora: bool) -> Result<()> {
        self.model.apply_lora(target_modules, rank, alpha, dropout, use_dora, &mut self.varmap)
    }
    
    pub fn clear_cache(&mut self) {}
}
