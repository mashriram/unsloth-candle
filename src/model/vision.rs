use candle_core::{DType, Device, Result, Tensor, Module};
use candle_nn::{Activation, VarBuilder, LayerNorm, Linear};

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub projection_dim: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_channels: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub layer_norm_eps: f64,
}

struct ClipVisionEmbeddings {
    patch_embedding: candle_nn::Conv2d,
    position_embedding: candle_nn::Embedding,
    class_embedding: Tensor,
    patch_size: usize,
    num_patches: usize,
    num_positions: usize,
}

impl ClipVisionEmbeddings {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let patch_size = cfg.patch_size;
        let num_patches = (cfg.image_size / patch_size) * (cfg.image_size / patch_size);
        let num_positions = num_patches + 1; // +1 for cls token

        let patch_embedding = candle_nn::conv2d_no_bias(
            cfg.num_channels,
            cfg.hidden_size,
            patch_size,
            candle_nn::Conv2dConfig {
                stride: patch_size,
                ..Default::default()
            },
            vb.pp("patch_embedding"),
        )?;
        
        let position_embedding = candle_nn::embedding(num_positions, cfg.hidden_size, vb.pp("position_embedding"))?;
        let class_embedding = vb.get((cfg.hidden_size,), "class_embedding")?;

        Ok(Self {
            patch_embedding,
            position_embedding,
            class_embedding,
            patch_size,
            num_patches,
            num_positions,
        })
    }

    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let batch_size = pixel_values.dim(0)?;
        
        // patch_embedding: [b, c, h, w] -> [b, embed_dim, grid, grid]
        let patch_embeds = self.patch_embedding.forward(pixel_values)?;
        let patch_embeds = patch_embeds.flatten_from(2)?.transpose(1, 2)?; // [b, num_patches, embed_dim]
        
        
        let class_embeds = self.class_embedding.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, embed_dim]
        let class_embeds = class_embeds.broadcast_as((batch_size, 1, self.class_embedding.dim(0)?))?;
        let embeddings = Tensor::cat(&[&class_embeds, &patch_embeds], 1)?; // [b, num_patches+1, embed_dim]
        
        let position_ids = Tensor::arange(0u32, self.num_positions as u32, pixel_values.device())?;
        let position_embeds = self.position_embedding.forward(&position_ids)?.unsqueeze(0)?;
        
        embeddings.broadcast_add(&position_embeds)
    }
}

struct ClipAttention {
    k_proj: Linear,
    v_proj: Linear,
    q_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    scale: f64,
    head_dim: usize,
}

impl ClipAttention {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let embed_dim = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let head_dim = embed_dim / num_heads;
        
        let k_proj = candle_nn::linear(embed_dim, embed_dim, vb.pp("k_proj"))?;
        let v_proj = candle_nn::linear(embed_dim, embed_dim, vb.pp("v_proj"))?;
        let q_proj = candle_nn::linear(embed_dim, embed_dim, vb.pp("q_proj"))?;
        let out_proj = candle_nn::linear(embed_dim, embed_dim, vb.pp("out_proj"))?;
        
        Ok(Self {
            k_proj,
            v_proj,
            q_proj,
            out_proj,
            num_heads,
            scale: (head_dim as f64).powf(-0.5),
            head_dim,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;
        
        let q = q.reshape((b, s, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let k = k.reshape((b, s, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let v = v.reshape((b, s, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        
        let attn = (q.matmul(&k.t()?)? * self.scale)?;
        let attn = candle_nn::ops::softmax(&attn, candle_core::D::Minus1)?;
        let out = attn.matmul(&v)?;
        
        let out = out.transpose(1, 2)?.reshape((b, s, ()))?;
        self.out_proj.forward(&out)
    }
}

struct ClipMLP {
    fc1: Linear,
    fc2: Linear,
    act: Activation,
}

impl ClipMLP {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let fc1 = candle_nn::linear(cfg.hidden_size, cfg.intermediate_size, vb.pp("fc1"))?;
        let fc2 = candle_nn::linear(cfg.intermediate_size, cfg.hidden_size, vb.pp("fc2"))?;
        Ok(Self {
            fc1,
            fc2,
            act: Activation::Gelu, // Fallback to Gelu
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(x)?;
        let x = self.act.forward(&x)?;
        self.fc2.forward(&x)
    }
}

struct ClipEncoderLayer {
    self_attn: ClipAttention,
    layer_norm1: LayerNorm,
    mlp: ClipMLP,
    layer_norm2: LayerNorm,
}

impl ClipEncoderLayer {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let self_attn = ClipAttention::load(vb.pp("self_attn"), cfg)?;
        let layer_norm1 = candle_nn::layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("layer_norm1"))?;
        let mlp = ClipMLP::load(vb.pp("mlp"), cfg)?;
        let layer_norm2 = candle_nn::layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("layer_norm2"))?;
        Ok(Self {
            self_attn,
            layer_norm1,
            mlp,
            layer_norm2,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x;
        let x = self.layer_norm1.forward(x)?;
        let x = self.self_attn.forward(&x)?;
        let x = (residual + x)?;
        
        let residual = &x;
        let x = self.layer_norm2.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        (residual + x)
    }
}

struct ClipEncoder {
    layers: Vec<ClipEncoderLayer>,
}

impl ClipEncoder {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(ClipEncoderLayer::load(vb.pp(&format!("layers.{}", i)), cfg)?);
        }
        Ok(Self { layers })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }
        Ok(x)
    }
}

pub struct ClipVisionTransformer {
    embeddings: ClipVisionEmbeddings,
    pre_layrnorm: LayerNorm,
    encoder: ClipEncoder,
    post_layernorm: LayerNorm,
}

impl ClipVisionTransformer {
     pub fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let embeddings = ClipVisionEmbeddings::load(vb.pp("embeddings"), cfg)?;
        let pre_layrnorm = candle_nn::layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("pre_layrnorm"))?;
        let encoder = ClipEncoder::load(vb.pp("encoder"), cfg)?;
        let post_layernorm = candle_nn::layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("post_layernorm"))?;
        
        Ok(Self {
            embeddings,
            pre_layrnorm,
            encoder,
            post_layernorm,
        })
    }
    
    pub fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let x = self.embeddings.forward(pixel_values)?;
        let x = self.pre_layrnorm.forward(&x)?;
        let x = self.encoder.forward(&x)?;
        let x = self.post_layernorm.forward(&x)?;
        // Return [b, s, h] (all tokens)
        // Usually we drop CLS or use CLS depending on model.
        // Llava uses all tokens (including CLS? or excluding? usually excluding CLS if doing patch merge, or including).
        // I'll return full sequence.
        Ok(x)
    }
}
