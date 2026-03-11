use candle_core::{DType, Device, Result};
use candle_nn::{VarBuilder, VarMap};
use candle_transformers::models::llama::LlamaEosToks;
use crate::model::RustModel;
use crate::model::llama::{Llama, Config};
use pyo3::prelude::*;
use std::path::PathBuf;

/// Parse common fields needed for LlamaConfig from raw JSON
fn parse_llama_config(
    json: &serde_json::Value,
    load_in_4bit: bool,
    use_gradient_checkpointing: bool,
) -> Result<Config> {
    let hidden_size = json["hidden_size"].as_u64().ok_or_else(|| candle_core::Error::Msg("missing hidden_size".into()))? as usize;
    let intermediate_size = json["intermediate_size"].as_u64().ok_or_else(|| candle_core::Error::Msg("missing intermediate_size".into()))? as usize;
    let vocab_size = json["vocab_size"].as_u64().ok_or_else(|| candle_core::Error::Msg("missing vocab_size".into()))? as usize;
    let num_hidden_layers = json["num_hidden_layers"].as_u64().ok_or_else(|| candle_core::Error::Msg("missing num_hidden_layers".into()))? as usize;
    let num_attention_heads = json["num_attention_heads"].as_u64().ok_or_else(|| candle_core::Error::Msg("missing num_attention_heads".into()))? as usize;
    let num_key_value_heads = json["num_key_value_heads"].as_u64().map(|v| v as usize).unwrap_or(num_attention_heads);
    let rms_norm_eps = json["rms_norm_eps"].as_f64().unwrap_or(1e-5);
    let rope_theta = json["rope_theta"].as_f64().unwrap_or(10000.0) as f32;
    let bos_token_id = json["bos_token_id"].as_u64().map(|v| v as u32);
    let eos_token_id = match json["eos_token_id"].as_u64() {
        Some(v) => Some(LlamaEosToks::Single(v as u32)),
        None => match json["eos_token_id"].as_array() {
            Some(arr) => {
                let toks: Vec<u32> = arr.iter().filter_map(|v| v.as_u64().map(|u| u as u32)).collect();
                if toks.is_empty() { None } else { Some(LlamaEosToks::Multiple(toks)) }
            },
            None => None,
        }
    };
    let max_position_embeddings = json["max_position_embeddings"].as_u64().unwrap_or(4096) as usize;
    let tie_word_embeddings = json["tie_word_embeddings"].as_bool().unwrap_or(false);
    let rope_scaling = json["rope_scaling"].as_object().cloned().map(|m| {
        let typ = m.get("type").and_then(|v| v.as_str()).unwrap_or("linear").to_string();
        let factor = m.get("factor").and_then(|v| v.as_f64()).unwrap_or(1.0);
        (typ, factor)
    });

    Ok(Config {
        hidden_size, intermediate_size, vocab_size,
        num_hidden_layers, num_attention_heads, num_key_value_heads,
        rms_norm_eps, rope_theta, bos_token_id, eos_token_id,
        max_position_embeddings, tie_word_embeddings,
        use_flash_attn: false,
        rope_scaling,
        use_gradient_checkpointing,
        load_in_4bit,
    })
}

pub fn load_model(
    model_name: &str,
    load_in_4bit: bool,
    use_gradient_checkpointing: bool,
    device: &Device
) -> Result<RustModel> {
    // 1. Download/Locate model using Python's huggingface_hub
    let model_dir = Python::with_gil(|py| -> PyResult<PathBuf> {
        let model_path = std::path::Path::new(model_name);
        if model_path.is_dir() {
            return Ok(model_path.to_path_buf());
        }

        let hf_hub = PyModule::import_bound(py, "huggingface_hub")?;
        let kwargs = pyo3::types::PyDict::new_bound(py);
        kwargs.set_item("repo_id", model_name)?;
        kwargs.set_item("allow_patterns", vec![
            "config.json", "*.safetensors", "model.safetensors.index.json",
            "*.json"  // for tokenizer config etc.
        ])?;
        let path: String = hf_hub.call_method("snapshot_download", (), Some(&kwargs))?.extract()?;
        Ok(PathBuf::from(path))
    }).map_err(|e| candle_core::Error::Msg(format!("Python hf_hub failed: {}", e)))?;

    // 2. Load Config JSON
    let config_filename = model_dir.join("config.json");
    let json: serde_json::Value = serde_json::from_reader(std::fs::File::open(config_filename)?)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

    // 3. Detect architecture
    let architectures = json["architectures"]
        .as_array()
        .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect::<Vec<_>>())
        .unwrap_or_else(|| vec!["LlamaForCausalLM".to_string()]);
    let arch = architectures.first().map(|s| s.as_str()).unwrap_or("LlamaForCausalLM");
    println!("Detected architecture: {}", arch);

    // 4. Determine dtype
    // CPU cannot do BF16/F16 matmul — always use F32
    // GPU can use the model's preferred dtype
    let dtype = if device.is_cpu() {
        DType::F32
    } else if load_in_4bit {
        DType::BF16
    } else {
        match json["torch_dtype"].as_str() {
            Some("bfloat16") => DType::BF16,
            Some("float16") => DType::F16,
            _ => DType::F32,
        }
    };
    println!("Using dtype: {:?}", dtype);

    // 5. Locate weight files
    let model_file = model_dir.join("model.safetensors");
    let filenames = if model_file.exists() {
        vec![model_file]
    } else {
        let index_file = model_dir.join("model.safetensors.index.json");
        let index: serde_json::Value = serde_json::from_reader(std::fs::File::open(index_file)?)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let weight_map = index.get("weight_map")
            .ok_or_else(|| candle_core::Error::Msg("Missing weight_map in index".into()))?
            .as_object()
            .ok_or_else(|| candle_core::Error::Msg("weight_map is not an object".into()))?;
        let mut files = std::collections::HashSet::new();
        for (_, v) in weight_map {
            if let Some(f) = v.as_str() { files.insert(f.to_string()); }
        }
        files.into_iter().map(|f| model_dir.join(f)).collect()
    };

    let varmap = VarMap::new();
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, device)? };

    // 6. Dispatch by architecture string
    match arch {
        // ─── Llama family (includes SmolLM which re-uses LlamaForCausalLM) ─────
        "LlamaForCausalLM" => {
            let config = parse_llama_config(&json, load_in_4bit, use_gradient_checkpointing)?;
            let model = Llama::load(vb, &config)?;
            Ok(RustModel::Llama(crate::model::LlamaModel::new(model, config, device.clone(), dtype, varmap)))
        }

        // ─── Mistral (proper sliding window, separate from Llama) ───────────────
        "MistralForCausalLM" | "MistralNemoForCausalLM" => {
            let cfg: crate::model::mistral::Config = serde_json::from_value(json)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            let model = crate::model::mistral::Mistral::load(vb, &cfg)?;
            Ok(RustModel::Mistral(crate::model::mistral::MistralModel::new(model, cfg, device.clone(), dtype, varmap)))
        }

        // ─── Mixtral MoE ─────────────────────────────────────────────────────────
        "MixtralForCausalLM" => {
            let cfg: crate::model::mixtral::Config = serde_json::from_value(json)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            let model = crate::model::mixtral::MixtralBody::load(vb, &cfg)?;
            Ok(RustModel::Mixtral(crate::model::mixtral::MixtralModel::new(model, cfg, device.clone(), dtype, varmap)))
        }

        // ─── Qwen2 ───────────────────────────────────────────────────────────────
        "Qwen2ForCausalLM" => {
            let cfg: crate::model::qwen2::Config = serde_json::from_value(json)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            let model = crate::model::qwen2::Qwen2::load(vb, &cfg)?;
            Ok(RustModel::Qwen2(crate::model::qwen2::Qwen2Model::new(model, cfg, device.clone(), dtype, varmap)))
        }

        // ─── Qwen3 dense ─────────────────────────────────────────────────────────
        "Qwen3ForCausalLM" => {
            let cfg: crate::model::qwen3::Config = serde_json::from_value(json)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            let model = crate::model::qwen3::Qwen3::load(vb, &cfg)?;
            Ok(RustModel::Qwen3(crate::model::qwen3::Qwen3Model::new(model, cfg, device.clone(), dtype, varmap)))
        }

        // ─── Qwen3-MoE ───────────────────────────────────────────────────────────
        "Qwen3MoeForCausalLM" => {
            let cfg: crate::model::qwen3_moe::Config = serde_json::from_value(json)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            let model = crate::model::qwen3_moe::Qwen3Moe::load(vb, &cfg)?;
            Ok(RustModel::Qwen3Moe(crate::model::qwen3_moe::Qwen3MoeModel::new(model, cfg, device.clone(), dtype, varmap)))
        }

        // ─── Qwen2-MoE ───────────────────────────────────────────────────────────
        "Qwen2MoeForCausalLM" => {
            let cfg: crate::model::qwen2_moe::Config = serde_json::from_value(json.clone())
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            let lm_head = crate::model::qwen2::linear(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?;
            let model = crate::model::qwen2_moe::Qwen2Moe::load(vb, &cfg)?;
            Ok(RustModel::Qwen2Moe(crate::model::qwen2_moe::Qwen2MoeModel::new(model, cfg, device.clone(), dtype, varmap, lm_head)))
        }

        // ─── Gemma 2 ────────────────────────────────────────────────────────────
        "Gemma2ForCausalLM" => {
            let cfg: crate::model::gemma::Config = serde_json::from_value(json)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            let model = crate::model::gemma::Gemma2::load(vb, &cfg)?;
            Ok(RustModel::Gemma(crate::model::gemma::Gemma2Model::new(model, cfg, device.clone(), dtype, varmap)))
        }

        // ─── Gemma 3 (and Gemma 1B which uses GemmaForCausalLM) ─────────────────
        "GemmaForCausalLM" | "Gemma3ForCausalLM" => {
            let cfg: crate::model::gemma3::Config = serde_json::from_value(json)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            let model = crate::model::gemma3::Gemma3::load(vb, &cfg)?;
            Ok(RustModel::Gemma3(crate::model::gemma3::Gemma3Model::new(model, cfg, device.clone(), dtype, varmap)))
        }

        // ─── Phi-3 ───────────────────────────────────────────────────────────────
        "Phi3ForCausalLM" => {
            let cfg: crate::model::phi3::Config = serde_json::from_value(json)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            let model = crate::model::phi3::Phi3::load(vb, &cfg)?;
            Ok(RustModel::Phi3(crate::model::phi3::Phi3Model::new(model, cfg, device.clone(), dtype, varmap)))
        }

        // ─── Phi-4 ───────────────────────────────────────────────────────────────
        "Phi4ForCausalLM" | "PhiMoEForCausalLM" => {
            let cfg: crate::model::phi4::Config = serde_json::from_value(json)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            let model = crate::model::phi4::Phi4::load(vb, &cfg)?;
            Ok(RustModel::Phi4(crate::model::phi4::Phi4Model::new(model, cfg, device.clone(), dtype, varmap)))
        }

        // ─── LLaVA / Pixtral (vision-language) ──────────────────────────────────
        "LlavaForConditionalGeneration"
        | "LlavaNextForConditionalGeneration"
        | "PixtralForConditionalGeneration" => {
            let cfg: crate::model::llava::Config = serde_json::from_value(json)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            let model = crate::model::llava::Llava::load(vb, &cfg)?;
            Ok(RustModel::Llava(crate::model::llava::LlavaModel::new(model, cfg, device.clone(), dtype, varmap)))
        }

        // ─── Qwen2-VL ────────────────────────────────────────────────────────────
        "Qwen2VLForConditionalGeneration" | "Qwen3VLForConditionalGeneration" => {
            let cfg: crate::model::qwen2_vl::Config = serde_json::from_value(json)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            let model = crate::model::qwen2_vl::Qwen2VL::load(vb, &cfg)?;
            Ok(RustModel::Qwen2VL(crate::model::qwen2_vl::Qwen2VLModel::new(model, cfg, device.clone(), dtype, varmap)))
        }

        // ─── GPT-NeoX / Pythia ──────────────────────────────────────────────────
        "GPTNeoXForCausalLM" => {
            let cfg: crate::model::gpt_neox::Config = serde_json::from_value(json)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            let model = crate::model::gpt_neox::GPTNeoX::load(vb, &cfg)?;
            Ok(RustModel::GPTNeoX(crate::model::gpt_neox::GPTNeoXModel::new(model, cfg, device.clone(), dtype, varmap)))
        }

        // ─── Cohere Command-R ────────────────────────────────────────────────────
        "CohereForCausalLM" | "Cohere2ForCausalLM" => {
            let cfg: crate::model::cohere::Config = serde_json::from_value(json)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            let model = crate::model::cohere::Cohere::load(vb, &cfg)?;
            Ok(RustModel::Cohere(crate::model::cohere::CohereModel::new(model, cfg, device.clone(), dtype, varmap)))
        }

        // ─── IBM Granite ─────────────────────────────────────────────────────────
        "GraniteForCausalLM" | "Granite3ForCausalLM" | "GraniteForCausalLMFamily" => {
            let cfg: crate::model::granite::Config = serde_json::from_value(json)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            let model = crate::model::granite::Granite::load(vb, &cfg)?;
            Ok(RustModel::Granite(crate::model::granite::GraniteModel::new(model, cfg, device.clone(), dtype, varmap)))
        }

        // ─── Sarvam MoE ──────────────────────────────────────────────────────────
        "SarvamMoEForCausalLM" => {
            let cfg: crate::model::sarvam_moe::Config = serde_json::from_value(json)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            let model = crate::model::sarvam_moe::SarvamMoe::load(vb, &cfg)?;
            Ok(RustModel::SarvamMoe(crate::model::sarvam_moe::SarvamMoeModel::new(model, cfg, device.clone(), dtype, varmap)))
        }

        // ─── OLMo 2 ──────────────────────────────────────────────────────────────
        "OlmoForCausalLM" | "Olmo2ForCausalLM" => {
            let cfg: crate::model::olmo::Config = serde_json::from_value(json)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            let model = crate::model::olmo::Olmo::load(vb, &cfg)?;
            Ok(RustModel::Olmo(crate::model::olmo::OlmoModel::new(model, cfg, device.clone(), dtype, varmap)))
        }

        // ─── StarCoder2 ──────────────────────────────────────────────────────────
        "Starcoder2ForCausalLM" => {
            let cfg: crate::model::starcoder2::Config = serde_json::from_value(json)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            let model = crate::model::starcoder2::StarCoder2::load(vb, &cfg)?;
            Ok(RustModel::StarCoder2(crate::model::starcoder2::StarCoder2Model::new(model, cfg, device.clone(), dtype, varmap)))
        }

        // ─── DeepSeek-V2 / V3 ────────────────────────────────────────────────────
        "DeepseekV2ForCausalLM" | "DeepseekV3ForCausalLM" => {
            let cfg: crate::model::deepseek_v2::Config = serde_json::from_value(json)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            let model = crate::model::deepseek_v2::DeepSeekV2::load(vb, &cfg)?;
            Ok(RustModel::DeepSeekV2(crate::model::deepseek_v2::DeepSeekV2Model::new(model, cfg, device.clone(), dtype, varmap)))
        }

        // ─── Unknown ─────────────────────────────────────────────────────────────
        _ => Err(candle_core::Error::Msg(format!(
            "Unsupported architecture: '{}'. Supported: LlamaForCausalLM, MistralForCausalLM, \
            MixtralForCausalLM, Qwen2ForCausalLM, Qwen3ForCausalLM, Qwen3MoeForCausalLM, \
            Qwen2MoeForCausalLM, Gemma2ForCausalLM, GemmaForCausalLM, Phi3ForCausalLM, \
            Phi4ForCausalLM, LlavaForConditionalGeneration, Qwen2VLForConditionalGeneration, \
            GPTNeoXForCausalLM, CohereForCausalLM, GraniteForCausalLM, SarvamMoEForCausalLM, \
            OlmoForCausalLM, Starcoder2ForCausalLM, DeepseekV2ForCausalLM", arch
        ))),
    }
}
