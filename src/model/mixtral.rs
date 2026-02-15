use candle_core::{DType, Device, Result, Tensor, Module, Var};
use candle_nn::{Activation, VarBuilder, RmsNorm, VarMap};
use crate::model::llama::{Cache}; // Reuse Cache as it is generic enough
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
    pub num_local_experts: usize,
    pub num_experts_per_tok: usize,
}

// Helper to create Linear layer (same as Llama)
fn linear(size1: usize, size2: usize, vb: VarBuilder) -> Result<AdapterLayer> {
    let weight = vb.get((size2, size1), "weight")?;
    let l = candle_nn::Linear::new(weight, None);
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
        let (_b, s, _h, _d) = x.dims4()?;
        let cos = self.cos.narrow(0, pos, seq_len)?;
        let sin = self.sin.narrow(0, pos, seq_len)?;
        candle_nn::rotary_emb::rope(x, &cos, &sin)
    }
}

pub struct MixtralAttention {
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

impl MixtralAttention {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
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

pub struct MixtralMLP {
    pub w1: AdapterLayer,
    pub w2: AdapterLayer,
    pub w3: AdapterLayer,
    pub act_fn: Activation,
}

impl MixtralMLP {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let intermediate_size = cfg.intermediate_size;
        
        let w1 = linear(hidden_size, intermediate_size, vb.pp("w1"))?;
        let w2 = linear(intermediate_size, hidden_size, vb.pp("w2"))?;
        let w3 = linear(hidden_size, intermediate_size, vb.pp("w3"))?;
        
        Ok(Self {
            w1,
            w2,
            w3,
            act_fn: Activation::Silu,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let lhs = self.w1.forward(x)?.apply(&self.act_fn)?;
        let rhs = self.w3.forward(x)?;
        self.w2.forward(&(lhs * rhs)?)
    }
}

pub struct SparseMoeBlock {
    gate: candle_nn::Linear,
    experts: Vec<MixtralMLP>,
    num_experts_per_tok: usize,
}

impl SparseMoeBlock {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let size = cfg.hidden_size;
        let gate = candle_nn::linear(size, cfg.num_local_experts, vb.pp("gate"))?;
        let mut experts = Vec::with_capacity(cfg.num_local_experts);
        for i in 0..cfg.num_local_experts {
            experts.push(MixtralMLP::load(vb.pp(&format!("experts.{}", i)), cfg)?);
        }
        Ok(Self {
            gate,
            experts,
            num_experts_per_tok: cfg.num_experts_per_tok,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [batch, seq, hidden]
        let (b, s, h) = x.dims3()?;
        let x_flat = x.reshape((b * s, h))?;
        
        // Router logits: [batch*seq, num_experts]
        let router_logits = self.gate.forward(&x_flat)?;
        let routing_weights = candle_nn::ops::softmax(&router_logits, candle_core::D::Minus1)?;
        
        // Select top-k
        // We generally use indices here.
        // candle has topk?
        // Note: topk returns (values, indices)
        // We need DType::F32
        
        // TODO: Optimized kernel for MoE routing would be better.
        // For now, we use a naive implementation which is slow but correct.
        // Or we iterate over experts.
        
        // Since we don't have a fused kernel easily available in candle-core without custom ops,
        // we can iterate over experts and mask? 
        // Or gather?

        // Standard TopK:
        // Use arg_sort_last_dim for workaround
        let indices_sorted = routing_weights.arg_sort_last_dim(true)?; // Descending
        let weights_sorted = routing_weights.gather(&indices_sorted, candle_core::D::Minus1)?;
        
        let routing_weights_topk = weights_sorted.narrow(candle_core::D::Minus1, 0, self.num_experts_per_tok)?;
        let selected_experts = indices_sorted.narrow(candle_core::D::Minus1, 0, self.num_experts_per_tok)?;
        
        // Re-normalize weights
        let sum_weights = routing_weights_topk.sum_keepdim(1)?;
        let routing_weights_topk = (routing_weights_topk / sum_weights)?; 
        
        // We need to route inputs to experts.
        // ... (comments removed for brevity)
        
        let mut final_output = Tensor::zeros_like(&x_flat)?;
        
        for k in 0..self.num_experts_per_tok {
             let expert_indices = selected_experts.narrow(1, k, 1)?.squeeze(1)?; // [b*s]
             let weights = routing_weights_topk.narrow(1, k, 1)?.squeeze(1)?; // [b*s]
             
             for (e_idx, expert) in self.experts.iter().enumerate() {
                 // Create mask: expert_indices == e_idx
                 let mask = expert_indices.eq(e_idx as u32)?; // [b*s] boolean 0/1
                 
                 // How to check if any? 
                 // We can simpler dispatch:
                 // Ideally we want index_select.
                 // candle doesn't have advanced boolean indexing like torch.
                 // We might need to use `gather` or `index_select`.
                 
                 // Alternative: Process entire batch for each expert and multiply by probability (which is 0 if not selected).
                 // But that defeats the sparse purpose (compute heavy).
                 
                 // Given this is "Unsloth" (Fast), we really should use a kernel or at least index_select.
                 
                 // Let's try to use `index_select`.
                 // We need indices where expert_indices == e_idx.
                 // This requires "nonzero" or "argwhere" which is missing/expensive in basic tensors without sync.
                 
                 // Fallback to dense simulation for correctness first?
                 // No, that's too slow (8x compute).
                 
                 // Let's try a simpler approach if batch size is small.
                 // Or just implement it naively with `where_cond`?
                 
                 // `out = expert(x_flat)`
                 // `final_output += out * weight * (expert_index == e_idx)`
                 // This uses 8x forward passes. Not ideal but strictly sparse.
                 
                 // WAIT. The standard implementation in Candle (e.g. Mixtral example) uses a gathering logic or simplified loop.
                 // Let's simulate the loop efficiently?
                 
                 // Let's assume we implement naive loop first to get it working, then optimize.
                 // "Go ahead test to the best"
                 
                 // Optimizing hint:
                 // We can use a trick:
                 // Only run forward pass on indices that match.
                 // But constructing the "subset tensor" requires `index_select`.
                 // We need the indices from the mask.
                 // `mask.nonzero()`? Candle doesn't expose it easily?
                 // We can read mask to CPU...
                 
                 // For now, let's process the loop.
                 // If we accept the slowdown, we can run all experts.
                 // But that's terrible.
                 
                 // What if `x_flat` is small?
                 
                 // Let's check if we can skip empty experts.
                 // We need to check if `mask` sums to > 0.
                 // `mask.sum_all()` -> scalar.
                 // .to_scalar::<u32>() > 0?
                 
                 // If mask is effectively 0s, we skip.
             }
        }
        
        // Okay, writing correct MoE routing without custom kernels in Rust/Candle is verbose.
        // I will use `candle_transformers::models::mixtral` approach as reference?
        // It likely uses `index_select`.
        
        // For the sake of this task, I will implement a simpler "Router" that iterates:
        // For each expert, find inputs assigned to it.
        // If > 0, forward.
        // Accumulate.
        
        // To do this efficiently:
        // We need `indices` of inputs for expert `e`.
        // We can't easily get variable length indices per expert without a sync.
        
        // STRATEGY: 
        // 1. Fetch `selected_experts` to CPU (sync). 
        // 2. Build buckets on CPU (Vec<Vec<usize>>).
        // 3. For each expert with items:
        //    a. `index_select` inputs from GPU x_flat.
        //    b. Run expert.
        //    c. `scatter_add` / `index_add` to output.
        // This involves CPU-GPU sync per step (or once per forward).
        // Once per forward is acceptable for inference/training loop control overhead.
        
        let selected_experts_cpu = selected_experts.to_vec2::<u32>()?; // [b*s, k]
        let routing_weights_cpu = routing_weights_topk.to_vec2::<f32>()?; // [b*s, k]
        
        // Re-organize on CPU
        let batch_size_flat = b * s;
        // Map: ExpertID -> Vec<(RowIndex, Weight)>
        let mut expert_map: Vec<Vec<(usize, f32)>> = vec![Vec::new(); self.experts.len()];
        
        for row in 0..batch_size_flat {
            for k in 0..self.num_experts_per_tok {
                 let e_idx = selected_experts_cpu[row][k] as usize;
                 let w = routing_weights_cpu[row][k];
                 expert_map[e_idx].push((row, w));
            }
        }
        
        for (e_idx, assignments) in expert_map.iter().enumerate() {
            if assignments.is_empty() { continue; }
            
            let indices: Vec<u32> = assignments.iter().map(|(r, _)| *r as u32).collect();
            let weights: Vec<f32> = assignments.iter().map(|(_, w)| *w).collect();
            
            let indices_len = indices.len();
            let indices_tensor = Tensor::from_vec(indices, (indices_len,), &x.device())?;
            
            let weights_len = weights.len();
            let weights_tensor = Tensor::from_vec(weights, (weights_len, 1), &x.device())?; // [N, 1]
            
            let expert_input = x_flat.index_select(&indices_tensor, 0)?; // [N, h]
            
            let expert_output = self.experts[e_idx].forward(&expert_input)?; // [N, h]
            let weighted_output = (expert_output * weights_tensor.broadcast_as((weights_tensor.dim(0)?, h))?)?;
            
            // Add back to final_output
            // index_add(dim, index, source) -> adds source to self at index
            final_output = final_output.index_add(&indices_tensor, &weighted_output, 0)?;
        }
        
        final_output.reshape((b, s, h))
    }
}

pub struct MixtralBlock {
    attn: MixtralAttention,
    block_sparse_moe: SparseMoeBlock,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl MixtralBlock {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let input_layernorm = RmsNorm::new(vb.pp("input_layernorm").get(cfg.hidden_size, "weight")?, cfg.rms_norm_eps);
        let attn = MixtralAttention::load(vb.pp("self_attn"), cfg)?;
        let post_attention_layernorm = RmsNorm::new(vb.pp("post_attention_layernorm").get(cfg.hidden_size, "weight")?, cfg.rms_norm_eps);
        let block_sparse_moe = SparseMoeBlock::load(vb.pp("block_sparse_moe"), cfg)?;
        
        Ok(Self {
            attn,
            block_sparse_moe,
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
        let x = self.block_sparse_moe.forward(&x)?;
        let x = (x + residual)?;
        Ok(x)
    }
}

pub struct MixtralModel {
    pub model: MixtralBody, // Wrapper around layers
    pub config: Config,
    pub device: Device,
    pub dtype: DType,
    pub cache: Cache,
    pub varmap: VarMap,
}

pub struct MixtralBody {
    pub embed_tokens: candle_nn::Embedding,
    pub layers: Vec<MixtralBlock>,
    pub norm: RmsNorm,
    pub lm_head: AdapterLayer,
}

impl MixtralBody {
    pub fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let embed_tokens = candle_nn::Embedding::new(vb.pp("model.embed_tokens").get((cfg.vocab_size, cfg.hidden_size), "weight")?, cfg.hidden_size);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(MixtralBlock::load(vb.pp(&format!("model.layers.{}", i)), cfg)?);
        }
        let norm = RmsNorm::new(vb.pp("model.norm").get(cfg.hidden_size, "weight")?, cfg.rms_norm_eps);
        let lm_head = linear(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?;
        
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
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
}

impl MixtralModel {
    pub fn new(model: MixtralBody, config: Config, device: Device, dtype: DType, varmap: VarMap) -> Self {
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
    
    pub fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
         let model = MixtralBody::load(vb.clone(), cfg)?;
         Ok(Self::new(model, cfg.clone(), vb.device().clone(), vb.dtype(), VarMap::new())) 
         // Note: varmap passed here should be external if we want to track vars?
         // In loader.rs we pass the varmap used to load.
         // Here `load` creates the Body. `Self` construction happens in loader.
    }

    pub fn forward(&mut self, input_ids: &Tensor, pos: usize) -> Result<Tensor> {
        self.model.forward(input_ids, pos, &mut self.cache)
    }

    pub fn clear_cache(&mut self) {
        self.cache = Cache::new(true, self.config.num_hidden_layers);
    }
    
    pub fn apply_lora(&mut self, target_modules: Vec<String>, rank: usize, alpha: f64, _dropout: f64, use_dora: bool) -> Result<()> {
         let scaling = alpha / rank as f64;
         let device = self.device.clone();
         let dtype = self.dtype;
        
         for (i, layer) in self.model.layers.iter_mut().enumerate() {
             use crate::model::inject_lora;

             // Attention
             if target_modules.contains(&"q_proj".to_string()) {
                inject_lora(&mut layer.attn.q_proj, rank, scaling, &mut self.varmap, &device, dtype, format!("model.layers.{}.self_attn.q_proj", i), use_dora)?;
             }
             if target_modules.contains(&"k_proj".to_string()) {
                inject_lora(&mut layer.attn.k_proj, rank, scaling, &mut self.varmap, &device, dtype, format!("model.layers.{}.self_attn.k_proj", i), use_dora)?;
             }
             if target_modules.contains(&"v_proj".to_string()) {
                inject_lora(&mut layer.attn.v_proj, rank, scaling, &mut self.varmap, &device, dtype, format!("model.layers.{}.self_attn.v_proj", i), use_dora)?;
             }
             if target_modules.contains(&"o_proj".to_string()) {
                inject_lora(&mut layer.attn.o_proj, rank, scaling, &mut self.varmap, &device, dtype, format!("model.layers.{}.self_attn.o_proj", i), use_dora)?;
             }
             
             // MoE Experts
             // Usually target_modules are "w1", "w2", "w3" which applies to ALL experts.
             // Or sometimes specific? "experts.0.w1"?
             // We assume "w1" means applied to all experts if present.
             
             for (e_idx, expert) in layer.block_sparse_moe.experts.iter_mut().enumerate() {
                 if target_modules.contains(&"w1".to_string()) || target_modules.contains(&"gate_proj".to_string()) {
                     // Mixtral calls them w1, w2, w3.
                     // w1 is gate_proj equivalent? Unsloth/HF mapping:
                     // w1 -> gate_proj
                     // w3 -> up_proj
                     // w2 -> down_proj
                     // But Mixtral uses w1, w2, w3 names in safetensors usually. 
                     // Let's support both naming conventions.
                     
                    inject_lora(&mut expert.w1, rank, scaling, &mut self.varmap, &device, dtype, format!("model.layers.{}.block_sparse_moe.experts.{}.w1", i, e_idx), use_dora)?;
                 }
                 if target_modules.contains(&"w2".to_string()) || target_modules.contains(&"down_proj".to_string()) {
                     inject_lora(&mut expert.w2, rank, scaling, &mut self.varmap, &device, dtype, format!("model.layers.{}.block_sparse_moe.experts.{}.w2", i, e_idx), use_dora)?;
                 }
                 if target_modules.contains(&"w3".to_string()) || target_modules.contains(&"up_proj".to_string()) {
                     inject_lora(&mut expert.w3, rank, scaling, &mut self.varmap, &device, dtype, format!("model.layers.{}.block_sparse_moe.experts.{}.w3", i, e_idx), use_dora)?;
                 }
             }
             
             // What about router (gate)?
         }
         Ok(())
    }
}
