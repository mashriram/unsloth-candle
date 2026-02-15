use candle_core::{DType, Device, Result};
use candle_nn::{VarBuilder, VarMap};
use candle_transformers::models::llama::LlamaEosToks;
use crate::model::RustModel;
use crate::model::llama::{Llama, Config};
use pyo3::prelude::*;
use std::path::PathBuf;

pub fn load_model(
    model_name: &str, 
    load_in_4bit: bool, 
    device: &Device
) -> Result<RustModel> {
    // 1. Download/Locate model using Python's huggingface_hub
    let model_dir = Python::with_gil(|py| -> PyResult<PathBuf> {
        let hf_hub = PyModule::import_bound(py, "huggingface_hub")?;
        let kwargs = pyo3::types::PyDict::new_bound(py);
        kwargs.set_item("repo_id", model_name)?;
        kwargs.set_item("allow_patterns", vec!["config.json", "*.safetensors", "model.safetensors.index.json"])?;
        
        let path: String = hf_hub.call_method("snapshot_download", (), Some(&kwargs))?.extract()?;
        Ok(PathBuf::from(path))
    }).map_err(|e| candle_core::Error::Msg(format!("Python hf_hub failed: {}", e)))?;

    // 2. Load Config
    let config_filename = model_dir.join("config.json");
    let json: serde_json::Value = serde_json::from_reader(std::fs::File::open(config_filename)?)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    
    // Manual construction
    let hidden_size = json["hidden_size"].as_u64().ok_or(candle_core::Error::Msg("missing hidden_size".to_string()))? as usize;
    let intermediate_size = json["intermediate_size"].as_u64().ok_or(candle_core::Error::Msg("missing intermediate_size".to_string()))? as usize;
    let vocab_size = json["vocab_size"].as_u64().ok_or(candle_core::Error::Msg("missing vocab_size".to_string()))? as usize;
    let num_hidden_layers = json["num_hidden_layers"].as_u64().ok_or(candle_core::Error::Msg("missing num_hidden_layers".to_string()))? as usize;
    let num_attention_heads = json["num_attention_heads"].as_u64().ok_or(candle_core::Error::Msg("missing num_attention_heads".to_string()))? as usize;
    let num_key_value_heads = json["num_key_value_heads"].as_u64().map(|v| v as usize).unwrap_or(num_attention_heads);
    let rms_norm_eps = json["rms_norm_eps"].as_f64().ok_or(candle_core::Error::Msg("missing rms_norm_eps".to_string()))?;
    let rope_theta = json["rope_theta"].as_f64().unwrap_or(10000.0) as f32;
    let bos_token_id = json["bos_token_id"].as_u64().map(|v| v as u32);
    let eos_token_id = match json["eos_token_id"].as_u64() {
        Some(v) => Some(LlamaEosToks::Single(v as u32)),
        None => match json["eos_token_id"].as_array() {
            Some(arr) => {
                let mut toks = Vec::new();
                for v in arr {
                    if let Some(u) = v.as_u64() {
                        toks.push(u as u32);
                    }
                }
                if toks.is_empty() { None } else { Some(LlamaEosToks::Multiple(toks)) }
            },
            None => None,
        }
    };
    let max_position_embeddings = json["max_position_embeddings"].as_u64().unwrap_or(4096) as usize;
    let tie_word_embeddings = json["tie_word_embeddings"].as_bool().unwrap_or(false);
    let rope_scaling = json["rope_scaling"].as_object().cloned().map(|m| {
        let typ = m["type"].as_str().unwrap_or("linear").to_string();
        let factor = m["factor"].as_f64().unwrap_or(1.0);
        (typ, factor)
    });

    let architectures = json["architectures"]
        .as_array()
        .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect::<Vec<_>>())
        .unwrap_or_else(|| vec!["LlamaForCausalLM".to_string()]);

    let config = Config {
        hidden_size,
        intermediate_size,
        vocab_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        rms_norm_eps,
        rope_theta,
        bos_token_id,
        eos_token_id,
        max_position_embeddings,
        tie_word_embeddings,
        use_flash_attn: false,
        rope_scaling,
    };

    // 3. Load Weights
    let model_file = model_dir.join("model.safetensors");
    let filenames = if model_file.exists() {
        vec![model_file]
    } else {
        let index_file = model_dir.join("model.safetensors.index.json");
        let index: serde_json::Value = serde_json::from_reader(std::fs::File::open(index_file)?)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        
        let weight_map = index.get("weight_map")
            .ok_or_else(|| candle_core::Error::Msg("Missing weight_map in index".to_string()))?
            .as_object()
            .ok_or_else(|| candle_core::Error::Msg("weight_map is not an object".to_string()))?;
        
        let mut files = std::collections::HashSet::new();
        for (_, v) in weight_map {
            if let Some(f) = v.as_str() {
                files.insert(f.to_string());
            }
        }
        
        let mut paths = Vec::new();
        for f in files {
            paths.push(model_dir.join(f));
        }
        paths
    };
    
    let dtype = if load_in_4bit {
        if device.is_cpu() {
            DType::F32
        } else {
            DType::BF16
        }
    } else {
        DType::F32
    };

    let varmap = VarMap::new();
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, device)? };
    
    // 4. Instantiate Model
    let arch = architectures.first().map(|s| s.as_str()).unwrap_or("LlamaForCausalLM");
    println!("Detected architecture: {}", arch);

    match arch {
        "LlamaForCausalLM" | "MistralForCausalLM" => {
            // ... existing Llama config parsing is reused ...
            // Wait, we constructed `config` object manually above *before* parsing architecture string?
            // Yes, lines 63-78. 
            // This `config` is `llama::Config`.
            let model = Llama::load(vb, &config)?;
            let llama_model = crate::model::LlamaModel::new(model, config, device.clone(), dtype, varmap);
            Ok(RustModel::Llama(llama_model))
        }
        "MixtralForCausalLM" => {
            let mixtral_config: crate::model::mixtral::Config = serde_json::from_value(json).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            let model = crate::model::mixtral::MixtralBody::load(vb, &mixtral_config)?;
            let mixtral_model = crate::model::mixtral::MixtralModel::new(model, mixtral_config, device.clone(), dtype, varmap);
            Ok(RustModel::Mixtral(mixtral_model))
        }
        "Qwen2ForCausalLM" => {
            let qwen2_config: crate::model::qwen2::Config = serde_json::from_value(json).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            let model = crate::model::qwen2::Qwen2::load(vb, &qwen2_config)?;
            let qwen2_model = crate::model::qwen2::Qwen2Model::new(model, qwen2_config, device.clone(), dtype, varmap);
            Ok(RustModel::Qwen2(qwen2_model))
        }
        "GemmaForCausalLM" | "Gemma2ForCausalLM" => {
            let gemma_config: crate::model::gemma::Config = serde_json::from_value(json).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            let model = crate::model::gemma::Gemma2::load(vb, &gemma_config)?;
            let gemma_model = crate::model::gemma::Gemma2Model::new(model, gemma_config, device.clone(), dtype, varmap);
            Ok(RustModel::Gemma(gemma_model))
        }
        "Phi3ForCausalLM" => {
             let phi3_config: crate::model::phi3::Config = serde_json::from_value(json).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
             let model = crate::model::phi3::Phi3::load(vb, &phi3_config)?;
             let phi3_model = crate::model::phi3::Phi3Model::new(model, phi3_config, device.clone(), dtype, varmap);
             Ok(RustModel::Phi3(phi3_model))
        }
        "LlavaForConditionalGeneration" | "LlavaNextForConditionalGeneration" | "PixtralForConditionalGeneration" => {
             let llava_config: crate::model::llava::Config = serde_json::from_value(json).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
             let model = crate::model::llava::Llava::load(vb, &llava_config)?;
             let llava_model = crate::model::llava::LlavaModel::new(model, llava_config, device.clone(), dtype, varmap);
             Ok(RustModel::Llava(llava_model))
        }
        "Qwen2VLForConditionalGeneration" => {
             let qwen2vl_config: crate::model::qwen2_vl::Config = serde_json::from_value(json).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
             let model = crate::model::qwen2_vl::Qwen2VL::load(vb, &qwen2vl_config)?;
             let qwen2vl_model = crate::model::qwen2_vl::Qwen2VLModel::new(model, qwen2vl_config, device.clone(), dtype, varmap);
             Ok(RustModel::Qwen2VL(qwen2vl_model))
        }
        "GPTNeoXForCausalLM" => {
             let gptneox_config: crate::model::gpt_neox::Config = serde_json::from_value(json).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
             let model = crate::model::gpt_neox::GPTNeoX::load(vb, &gptneox_config)?;
             let gptneox_model = crate::model::gpt_neox::GPTNeoXModel::new(model, gptneox_config, device.clone(), dtype, varmap);
             Ok(RustModel::GPTNeoX(gptneox_model))
        }
        "CohereForCausalLM" => {
             let cohere_config: crate::model::cohere::Config = serde_json::from_value(json).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
             let model = crate::model::cohere::Cohere::load(vb, &cohere_config)?;
             let cohere_model = crate::model::cohere::CohereModel::new(model, cohere_config, device.clone(), dtype, varmap);
             Ok(RustModel::Cohere(cohere_model))
        }
        "Qwen2MoeForCausalLM" => {
             let qwen2moe_config: crate::model::qwen2_moe::Config = serde_json::from_value(json).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
             // Qwen2MoE generally requires loading Qwen2-like weights but with MoE layers
             // Usually use same Qwen2Moe wrapper
             // We need to pass lm_head separately if not part of model structure?
             // Qwen2Moe generally includes lm_head in model... but our struct separates it usually?
             // Let's check Qwen2Moe implementation... it has lm_head in AdapterLayer in struct
             // Wait, in qwen2_moe.rs I didn't add lm_head to `Qwen2Moe` struct, only `embed`, `layers`, `norm`.
             // I added `lm_head` to `Qwen2MoeModel` wrapper.
             // So we need to load it here.
             let lm_head = crate::model::qwen2::linear(qwen2moe_config.hidden_size, qwen2moe_config.vocab_size, vb.pp("lm_head"))?;
             
             let model = crate::model::qwen2_moe::Qwen2Moe::load(vb, &qwen2moe_config)?;
             let qwen2moe_model = crate::model::qwen2_moe::Qwen2MoeModel::new(model, qwen2moe_config, device.clone(), dtype, varmap, lm_head);
             Ok(RustModel::Qwen2Moe(qwen2moe_model))
        }
        _ => Err(candle_core::Error::Msg(format!("Unsupported architecture: {}", arch))),
    }
}
