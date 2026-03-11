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
// New models
use crate::model::mistral::MistralModel;
use crate::model::qwen3::Qwen3Model;
use crate::model::qwen3_moe::Qwen3MoeModel;
use crate::model::gemma3::Gemma3Model;
use crate::model::granite::GraniteModel;
use crate::model::olmo::OlmoModel;
use crate::model::starcoder2::StarCoder2Model;
use crate::model::phi4::Phi4Model;
use crate::model::sarvam_moe::SarvamMoeModel;
use crate::model::deepseek_v2::DeepSeekV2Model;

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
// New model modules
pub mod mistral;
pub mod qwen3;
pub mod qwen3_moe;
pub mod gemma3;
pub mod granite;
pub mod olmo;
pub mod starcoder2;
pub mod phi4;
pub mod sarvam_moe;
pub mod deepseek_v2;

pub use self::layers::{AdapterLayer, LoRALinear, DoRALinear};

pub enum RustModel {
    // Original models
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
    // New models
    Mistral(MistralModel),
    Qwen3(Qwen3Model),
    Qwen3Moe(Qwen3MoeModel),
    Gemma3(Gemma3Model),
    Granite(GraniteModel),
    Olmo(OlmoModel),
    StarCoder2(StarCoder2Model),
    Phi4(Phi4Model),
    SarvamMoe(SarvamMoeModel),
    DeepSeekV2(DeepSeekV2Model),
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
            Self::Mistral(m) => m.forward(input_ids, pos),
            Self::Qwen3(m) => m.forward(input_ids, pos),
            Self::Qwen3Moe(m) => m.forward(input_ids, pos),
            Self::Gemma3(m) => m.forward(input_ids, pos),
            Self::Granite(m) => m.forward(input_ids, pos),
            Self::Olmo(m) => m.forward(input_ids, pos),
            Self::StarCoder2(m) => m.forward(input_ids, pos),
            Self::Phi4(m) => m.forward(input_ids, pos),
            Self::SarvamMoe(m) => m.forward(input_ids, pos),
            Self::DeepSeekV2(m) => m.forward(input_ids, pos),
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
            Self::Mistral(m) => m.apply_lora(target_modules, rank, alpha, dropout, use_dora),
            Self::Qwen3(m) => m.apply_lora(target_modules, rank, alpha, dropout, use_dora),
            Self::Qwen3Moe(m) => m.apply_lora(target_modules, rank, alpha, dropout, use_dora),
            Self::Gemma3(m) => m.apply_lora(target_modules, rank, alpha, dropout, use_dora),
            Self::Granite(m) => m.apply_lora(target_modules, rank, alpha, dropout, use_dora),
            Self::Olmo(m) => m.apply_lora(target_modules, rank, alpha, dropout, use_dora),
            Self::StarCoder2(m) => m.apply_lora(target_modules, rank, alpha, dropout, use_dora),
            Self::Phi4(m) => m.apply_lora(target_modules, rank, alpha, dropout, use_dora),
            Self::SarvamMoe(m) => m.apply_lora(target_modules, rank, alpha, dropout, use_dora),
            Self::DeepSeekV2(m) => m.apply_lora(target_modules, rank, alpha, dropout, use_dora),
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
            Self::Mistral(m) => m.clear_cache(),
            Self::Qwen3(m) => m.clear_cache(),
            Self::Qwen3Moe(m) => m.clear_cache(),
            Self::Gemma3(m) => m.clear_cache(),
            Self::Granite(m) => m.clear_cache(),
            Self::Olmo(m) => m.clear_cache(),
            Self::StarCoder2(m) => m.clear_cache(),
            Self::Phi4(m) => m.clear_cache(),
            Self::SarvamMoe(m) => m.clear_cache(),
            Self::DeepSeekV2(m) => m.clear_cache(),
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
            Self::Mistral(m) => &m.device,
            Self::Qwen3(m) => &m.device,
            Self::Qwen3Moe(m) => &m.device,
            Self::Gemma3(m) => &m.device,
            Self::Granite(m) => &m.device,
            Self::Olmo(m) => &m.device,
            Self::StarCoder2(m) => &m.device,
            Self::Phi4(m) => &m.device,
            Self::SarvamMoe(m) => &m.device,
            Self::DeepSeekV2(m) => &m.device,
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
            Self::Mistral(m) => &m.varmap,
            Self::Qwen3(m) => &m.varmap,
            Self::Qwen3Moe(m) => &m.varmap,
            Self::Gemma3(m) => &m.varmap,
            Self::Granite(m) => &m.varmap,
            Self::Olmo(m) => &m.varmap,
            Self::StarCoder2(m) => &m.varmap,
            Self::Phi4(m) => &m.varmap,
            Self::SarvamMoe(m) => &m.varmap,
            Self::DeepSeekV2(m) => &m.varmap,
        }
    }
}

// ─── LlamaModel wrapper (kept here for backwards compatibility) ───────────────

pub struct LlamaModel {
    pub model: Llama,
    pub config: LlamaConfig,
    pub device: Device,
    pub dtype: DType,
    pub cache: Cache,
    pub varmap: VarMap,
    pub bias: Var,
}

impl LlamaModel {
    pub fn new(model: Llama, config: LlamaConfig, device: Device, dtype: DType, mut varmap: VarMap) -> Self {
        let cache = Cache::new(true, config.num_hidden_layers);
        let vocab_size = config.vocab_size;
        let _ = varmap.get(
            (vocab_size,),
            "trainable_bias",
            candle_nn::init::DEFAULT_KAIMING_NORMAL,
            dtype,
            &device
        ).unwrap();
        let bias = {
            let data = varmap.data().lock().unwrap();
            data.get("trainable_bias").expect("Bias should exist").clone()
        };
        Self { model, config, device, dtype, cache, varmap, bias }
    }

    pub fn forward(&mut self, input_ids: &Tensor, pos: usize) -> Result<Tensor> {
        let logits = self.model.forward(input_ids, pos, &mut self.cache)?;
        logits.broadcast_add(&self.bias.as_tensor())
    }

    pub fn clear_cache(&mut self) {
        self.cache = Cache::new(true, self.config.num_hidden_layers);
    }

    pub fn apply_lora(&mut self, target_modules: Vec<String>, rank: usize, alpha: f64, dropout: f64, use_dora: bool) -> Result<()> {
        self.model.apply_lora(target_modules, rank, alpha, dropout, use_dora, &mut self.varmap)
    }
}

// ─── inject_lora — shared LoRA/DoRA injection helper ─────────────────────────

pub fn inject_lora(
    layer: &mut AdapterLayer,
    rank: usize,
    scaling: f64,
    varmap: &mut VarMap,
    device: &Device,
    dtype: DType,
    prefix: String,
    use_dora: bool,
) -> Result<()> {
    let (out_dim, in_dim) = match layer {
        AdapterLayer::Linear(l) => l.weight().dims2()?,
        AdapterLayer::Linear4bit(l) => (l.out_features, l.in_features),
        _ => return Ok(()), // Already LoRA/DoRA
    };

    let lora_a = varmap.get((rank, in_dim), &format!("{}.lora_a", prefix), candle_nn::init::DEFAULT_KAIMING_NORMAL, dtype, device)?;
    let lora_b = varmap.get((out_dim, rank), &format!("{}.lora_b", prefix), candle_nn::init::ZERO, dtype, device)?;
    let base_layer = layer.clone();

    if use_dora {
        let w = base_layer.get_weight_f32()?;
        let m_init = w.sqr()?.sum_keepdim(1)?.sqrt()?.flatten_all()?;
        let m_name = format!("{}.lora_magnitude_vector", prefix);
        {
            let mut data = varmap.data().lock().unwrap();
            if !data.contains_key(&m_name) {
                data.insert(m_name.clone(), Var::from_tensor(&m_init)?);
            }
        }
        let m = varmap.get((out_dim,), &m_name, candle_nn::init::ZERO, dtype, device)?;
        *layer = AdapterLayer::DoRA(DoRALinear::new(base_layer, lora_a, lora_b, m, scaling));
    } else {
        *layer = AdapterLayer::LoRA(LoRALinear::new(base_layer, lora_a, lora_b, scaling));
    }
    Ok(())
}
