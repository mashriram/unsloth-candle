use candle_core::{DType, Device, Result, Tensor, Var};
use candle_nn::VarMap;

pub use crate::model::llama::{Llama, Cache, Config as LlamaConfig};
use crate::model::mixtral::MixtralModel;
use crate::model::qwen2::Qwen2Model;
use crate::model::qwen2_vl::Qwen2VLModel;
use crate::model::gpt_neox::GPTNeoXModel;
use crate::model::cohere::CohereModel;
use crate::model::qwen2_moe::Qwen2MoeModel;
use crate::model::gemma::Gemma2Model;
use crate::model::phi3::Phi3Model;
use crate::model::llava::LlavaModel;

pub mod layers;
pub mod linear4bit;
pub mod llama;
pub mod mixtral; 
pub mod qwen2;
pub mod gemma;
pub mod phi3;
pub mod vision;
pub mod llava;
pub mod qwen2_vl;
pub mod gpt_neox;
pub mod cohere;
pub mod qwen2_moe;

pub use self::layers::{AdapterLayer, LoRALinear, DoRALinear};

pub enum RustModel {
    Llama(LlamaModel),
    Mixtral(MixtralModel),
    Qwen2(Qwen2Model),
    Gemma(Gemma2Model),
    Phi3(Phi3Model),
    Llava(LlavaModel),
    Qwen2VL(Qwen2VLModel),
    GPTNeoX(GPTNeoXModel),
    Cohere(CohereModel),
    Qwen2Moe(Qwen2MoeModel),
}

impl RustModel {
    pub fn forward(&mut self, input_ids: &Tensor, pixel_values: Option<&Tensor>, pos: usize) -> Result<Tensor> {
        match self {
            Self::Llama(m) => m.forward(input_ids, pos),
            Self::Mixtral(m) => m.forward(input_ids, pos),
            Self::Qwen2(m) => m.forward(input_ids, pos),
            Self::Gemma(m) => m.forward(input_ids, pos),
            Self::Phi3(m) => m.forward(input_ids, pos),
            Self::Llava(m) => m.forward(input_ids, pixel_values, pos),
            Self::Qwen2VL(m) => m.forward(input_ids, pixel_values, pos),
            Self::GPTNeoX(m) => m.forward(input_ids, pos),
            Self::Cohere(m) => m.forward(input_ids, pos),
            Self::Qwen2Moe(m) => m.forward(input_ids, pos),
        }
    }
    
    pub fn apply_lora(&mut self, target_modules: Vec<String>, rank: usize, alpha: f64, dropout: f64, use_dora: bool) -> Result<()> {
        match self {
            Self::Llama(m) => m.apply_lora(target_modules, rank, alpha, dropout, use_dora),
            Self::Mixtral(m) => m.apply_lora(target_modules, rank, alpha, dropout, use_dora),
            Self::Qwen2(m) => m.apply_lora(target_modules, rank, alpha, dropout, use_dora),
            Self::Gemma(m) => m.apply_lora(target_modules, rank, alpha, dropout, use_dora),
            Self::Phi3(m) => m.apply_lora(target_modules, rank, alpha, dropout, use_dora),
            Self::Llava(m) => m.apply_lora(target_modules, rank, alpha, dropout, use_dora),
            Self::Qwen2VL(m) => m.apply_lora(target_modules, rank, alpha, dropout, use_dora),
            Self::GPTNeoX(m) => m.apply_lora(target_modules, rank, alpha, dropout, use_dora),
            Self::Cohere(m) => m.apply_lora(target_modules, rank, alpha, dropout, use_dora),
            Self::Qwen2Moe(m) => m.apply_lora(target_modules, rank, alpha, dropout, use_dora),
        }
    }
    
    pub fn clear_cache(&mut self) {
        match self {
            Self::Llama(m) => m.clear_cache(),
            Self::Mixtral(m) => m.clear_cache(),
            Self::Qwen2(m) => m.clear_cache(),
            Self::Gemma(m) => m.clear_cache(),
            Self::Phi3(m) => m.clear_cache(),
            Self::Llava(m) => m.clear_cache(),
            Self::Qwen2VL(m) => m.clear_cache(),
            Self::GPTNeoX(m) => m.clear_cache(),
            Self::Cohere(m) => m.clear_cache(),
            Self::Qwen2Moe(m) => m.clear_cache(),
        }
    }
    
    pub fn device(&self) -> &Device {
        match self {
            Self::Llama(m) => &m.device,
            Self::Mixtral(m) => &m.device,
            Self::Qwen2(m) => &m.device,
            Self::Gemma(m) => &m.device,
            Self::Phi3(m) => &m.device,
            Self::Llava(m) => &m.device,
            Self::Qwen2VL(m) => &m.device,
            Self::GPTNeoX(m) => &m.device,
            Self::Cohere(m) => &m.device,
            Self::Qwen2Moe(m) => &m.device,
        }
    }

    pub fn varmap(&self) -> &VarMap {
         match self {
             Self::Llama(m) => &m.varmap,
             Self::Mixtral(m) => &m.varmap,
             Self::Qwen2(m) => &m.varmap,
             Self::Gemma(m) => &m.varmap,
             Self::Phi3(m) => &m.varmap,
             Self::Llava(m) => &m.varmap,
             Self::Qwen2VL(m) => &m.varmap,
             Self::GPTNeoX(m) => &m.varmap,
             Self::Cohere(m) => &m.varmap,
             Self::Qwen2Moe(m) => &m.varmap,
         }
    }
}

// LlamaModel definition (kept here for now, or moved to llama.rs?)
// To keep things clean, LlamaModel struct should ideally be in llama.rs, but it references Config and Llama.
// Currently it was defined in mod.rs in previous steps. I should keep it here to avoid breaking changes unless I move it.

pub struct LlamaModel {
    pub model: Llama,
    pub config: LlamaConfig,
    pub device: Device,
    pub dtype: DType,
    pub cache: Cache,
    pub varmap: VarMap,
    pub bias: Var, // Trainable bias for verification
}

impl LlamaModel {
    pub fn new(model: Llama, config: LlamaConfig, device: Device, dtype: DType, mut varmap: VarMap) -> Self {
        let cache = Cache::new(true, config.num_hidden_layers);
        // Create a trainable bias.
        // We must add it to varmap to be picked up by optimizer.
        let vocab_size = config.vocab_size;
        let shape = (vocab_size,);
        
        // Ensure var is created (returns Tensor, so we ignore it)
        // If it already exists (e.g. from previous load), get will return it.
        // But get with init will try to init.
        // We use a safe check? Or just unwrap?
        // varmap.get will return existing if present.
        
        let _ = varmap.get(
            shape, 
            "trainable_bias", 
            candle_nn::init::DEFAULT_KAIMING_NORMAL, 
            dtype, 
            &device
        ).unwrap();
        
        // Retrieve the actual Var object
        let bias = {
             let data = varmap.data().lock().unwrap();
             data.get("trainable_bias").expect("Bias should exist").clone()
        };

        Self {
            model,
            config,
            device,
            dtype,
            cache,
            varmap,
            bias,
        }
    }

    pub fn forward(&mut self, input_ids: &Tensor, pos: usize) -> Result<Tensor> {
        let logits = self.model.forward(input_ids, pos, &mut self.cache)?;
        // Add bias (broadcasted)
        logits.broadcast_add(&self.bias.as_tensor())
    }

    pub fn clear_cache(&mut self) {
        self.cache = Cache::new(true, self.config.num_hidden_layers);
    }
    
    pub fn apply_lora(&mut self, target_modules: Vec<String>, rank: usize, alpha: f64, dropout: f64, use_dora: bool) -> Result<()> {
        self.model.apply_lora(target_modules, rank, alpha, dropout, use_dora, &mut self.varmap)
    }
}

pub fn inject_lora(layer: &mut AdapterLayer, rank: usize, scaling: f64, varmap: &mut VarMap, device: &Device, dtype: DType, prefix: String, use_dora: bool) -> Result<()> {
    // Check if we can apply LoRA
    let (out_dim, in_dim) = match layer {
        AdapterLayer::Linear(l) => {
            let w = l.weight();
            w.dims2()?
        },
        AdapterLayer::Linear4bit(l) => {
            (l.out_features, l.in_features) // No need to dequantize just for dims
        },
        _ => return Ok(()), // Already LoRA or other
    };

    let lora_a = varmap.get((rank, in_dim), &format!("{}.lora_a", prefix), candle_nn::init::DEFAULT_KAIMING_NORMAL, dtype, device)?;
    let lora_b = varmap.get((out_dim, rank), &format!("{}.lora_b", prefix), candle_nn::init::ZERO, dtype, device)?;
    
    // We need to clone the current layer to wrap it.
    // layer is &mut AdapterLayer. We can clone it.
    let base_layer = layer.clone();

    if use_dora {
        // DoRA needs magnitude vector initialization.
        // We need dequantized weights for this!
        let w = base_layer.get_weight_f32()?; // Uses our new helper
        let m_init = w.sqr()?.sum_keepdim(1)?.sqrt()?.flatten_all()?;
        
        let m_name = format!("{}.lora_magnitude_vector", prefix);
        {
            let mut data = varmap.data().lock().unwrap();
            if !data.contains_key(&m_name) {
                let m_var = Var::from_tensor(&m_init)?;
                data.insert(m_name.clone(), m_var);
            }
        }
        let m = varmap.get((out_dim,), &m_name, candle_nn::init::ZERO, dtype, device)?; 
        
        // Construct DoRA
        let new_layer = DoRALinear::new(base_layer, lora_a, lora_b, m, scaling);
        *layer = AdapterLayer::DoRA(new_layer);
    } else {
        // Construct LoRA
        let new_layer = LoRALinear::new(base_layer, lora_a, lora_b, scaling);
        *layer = AdapterLayer::LoRA(new_layer);
    }
    Ok(())
}
