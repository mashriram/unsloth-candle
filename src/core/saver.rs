//! Model save utilities: HF merged safetensors, GGUF, and 4-bit NF4 safetensors
//!
//! Three save modes:
//! 1. save_pretrained_merged  — Reads original safetensors + merges LoRA/DoRA, saves as HF safetensors
//! 2. save_to_gguf            — Converts merged weights to GGUF format via llama.cpp's convert script
//! 3. save_in_4bit            — Saves in NF4 4-bit quantized safetensors format
use candle_core::{DType, Result, Tensor};
use candle_nn::VarMap;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Read all tensors from the original safetensors files in model_dir.
fn load_base_tensors(model_dir: &Path, device: &candle_core::Device) -> Result<HashMap<String, Tensor>> {
    let mut tensors = HashMap::new();

    // Find safetensors files
    let model_file = model_dir.join("model.safetensors");
    let filenames: Vec<PathBuf> = if model_file.exists() {
        vec![model_file]
    } else {
        let index_file = model_dir.join("model.safetensors.index.json");
        if !index_file.exists() {
            return Err(candle_core::Error::Msg(
                format!("No safetensors found in {}", model_dir.display())
            ));
        }
        let index: serde_json::Value = serde_json::from_reader(
            std::fs::File::open(&index_file)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?
        ).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let weight_map = index.get("weight_map")
            .and_then(|v| v.as_object())
            .ok_or_else(|| candle_core::Error::Msg("Missing weight_map".into()))?;
        let mut files = std::collections::HashSet::new();
        for (_, v) in weight_map {
            if let Some(f) = v.as_str() { files.insert(f.to_string()); }
        }
        files.into_iter().map(|f| model_dir.join(f)).collect()
    };

    // Load all tensors from all safetensors files
    for file in &filenames {
        let buffer = std::fs::read(file).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let st = safetensors::SafeTensors::deserialize(&buffer)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        
        let t = candle_core::safetensors::load(file, device)?;
        for (k, v) in t {
            tensors.insert(k, v);
        }
    }

    Ok(tensors)
}

/// Get LoRA adapter tensors from VarMap (lora_a, lora_b, magnitude_vector).
fn get_lora_tensors(varmap: &VarMap) -> HashMap<String, Tensor> {
    let data = varmap.data().lock().unwrap();
    data.iter()
        .filter(|(k, _)| k.contains("lora_") || k.contains("magnitude_vector"))
        .map(|(k, v)| (k.clone(), v.as_tensor().clone()))
        .collect()
}

/// Merge a single LoRA adapter into a base weight.
/// LoRA: W_merged = W_base + (B @ A) * scaling
fn merge_lora_into_weight(
    base: &Tensor,
    lora_a: &Tensor,
    lora_b: &Tensor,
    scaling: f64,
) -> Result<Tensor> {
    // lora_a: [rank, in_dim], lora_b: [out_dim, rank]
    let delta = lora_b.matmul(lora_a)?;
    let scaled_delta = (delta * scaling)?;
    // Cast base to F32 if needed for the merge
    let base_f32 = base.to_dtype(DType::F32)?;
    let merged = (base_f32 + scaled_delta.to_dtype(DType::F32)?)?;
    Ok(merged)
}

/// Quantize a weight tensor to NF4 4-bit packed format.
/// Returns (packed_u8, scales).
fn quantize_to_nf4(weight: &Tensor, block_size: usize) -> Result<(Tensor, Tensor)> {
    let device = weight.device();
    let (out_features, in_features) = weight.dims2()?;
    let weight_cpu = weight.to_device(&candle_core::Device::Cpu)?.to_dtype(DType::F32)?;
    let weight_data = weight_cpu.flatten_all()?.to_vec1::<f32>()?;

    let nf4_values: [f32; 16] = [
        -1.0, -0.6961928, -0.5250730, -0.3949175,
        -0.2844414, -0.1847734, -0.0910500, 0.0,
        0.0795803, 0.1609302, 0.2461123, 0.3379152,
        0.4407098, 0.5626170, 0.7229568, 1.0,
    ];

    let num_blocks = weight_data.len().div_ceil(block_size);
    let mut scales = Vec::with_capacity(num_blocks);
    let mut packed = Vec::with_capacity(weight_data.len() / 2);

    for chunk in weight_data.chunks(block_size) {
        let max_val = chunk.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        scales.push(max_val);

        for pair in chunk.chunks(2) {
            if pair.len() == 2 {
                let q0 = quantize_closest(pair[0], max_val, &nf4_values);
                let q1 = quantize_closest(pair[1], max_val, &nf4_values);
                packed.push((q0 & 0x0F) | ((q1 & 0x0F) << 4));
            }
        }
    }

    let packed_tensor = Tensor::from_vec(packed, (out_features, in_features / 2), device)?;
    let scales_tensor = Tensor::from_vec(scales, (num_blocks,), device)?;
    Ok((packed_tensor, scales_tensor))
}

fn quantize_closest(v: f32, absmax: f32, nf4_values: &[f32]) -> u8 {
    if absmax == 0.0 { return 7; }
    let normalized = v / absmax;
    let mut best_idx = 0u8;
    let mut min_err = f32::MAX;
    for (i, &nf4) in nf4_values.iter().enumerate() {
        let err = (normalized - nf4).abs();
        if err < min_err { min_err = err; best_idx = i as u8; }
    }
    best_idx
}

// ─── Public save functions ──────────────────────────────────────────────────

/// Save merged model weights as HuggingFace safetensors.
/// Reads original weights from model_dir, merges LoRA/DoRA from varmap, saves to output_dir.
pub fn save_pretrained_merged(
    model_dir: &Path,
    varmap: &VarMap,
    config_json: &serde_json::Value,
    output_dir: &Path,
) -> Result<PathBuf> {
    std::fs::create_dir_all(output_dir)
        .map_err(|e| candle_core::Error::Msg(format!("mkdir failed: {}", e)))?;

    println!("  Reading original weights from {}...", model_dir.display());
    let device = candle_core::Device::Cpu;
    let mut tensors = load_base_tensors(model_dir, &device)?;
    println!("  Loaded {} base tensors", tensors.len());

    // Merge LoRA adapters into base weights
    let lora_tensors = get_lora_tensors(varmap);
    if !lora_tensors.is_empty() {
        println!("  Found {} LoRA adapter tensors, merging...", lora_tensors.len());

        // Group LoRA pairs: model.layers.X.self_attn.q_proj.lora_a + .lora_b
        let mut lora_pairs: HashMap<String, (Option<Tensor>, Option<Tensor>)> = HashMap::new();
        for (key, tensor) in &lora_tensors {
            let base_key = key.replace(".lora_a", "").replace(".lora_b", "");
            let entry = lora_pairs.entry(base_key).or_insert((None, None));
            if key.ends_with(".lora_a") {
                entry.0 = Some(tensor.clone());
            } else if key.ends_with(".lora_b") {
                entry.1 = Some(tensor.clone());
            }
        }

        for (base_key, (lora_a, lora_b)) in &lora_pairs {
            if let (Some(a), Some(b)) = (lora_a, lora_b) {
                // Find the corresponding base weight: try "base_key.weight"
                let weight_key = format!("{}.weight", base_key);
                if let Some(base_weight) = tensors.get(&weight_key) {
                    let rank = a.dim(0)?;
                    let scaling = 1.0; // scaling is already baked into lora_b during training
                    let merged = merge_lora_into_weight(base_weight, a, b, scaling)?;
                    tensors.insert(weight_key.clone(), merged);
                    println!("    ✓ Merged LoRA into {} (rank={})", weight_key, rank);
                }
            }
        }
    }

    // Save as safetensors (convert all to F32)
    let safetensors_path = output_dir.join("model.safetensors");
    let tensors_f32: HashMap<String, Tensor> = tensors.into_iter()
        .map(|(k, v)| {
            let v_f32 = v.to_dtype(DType::F32).unwrap_or(v);
            (k, v_f32)
        })
        .collect();
    let num_tensors = tensors_f32.len();
    candle_core::safetensors::save(&tensors_f32, &safetensors_path)?;

    // Save config.json
    let config_path = output_dir.join("config.json");
    let config_str = serde_json::to_string_pretty(config_json)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    std::fs::write(&config_path, config_str)
        .map_err(|e| candle_core::Error::Msg(format!("write config.json failed: {}", e)))?;

    println!("✓ Merged model saved to {}", output_dir.display());
    println!("  - model.safetensors ({} tensors)", num_tensors);
    println!("  - config.json");
    Ok(safetensors_path)
}

/// Save merged weights in 4-bit NF4 quantized safetensors format.
pub fn save_in_4bit(
    model_dir: &Path,
    varmap: &VarMap,
    config_json: &serde_json::Value,
    output_dir: &Path,
    block_size: usize,
) -> Result<PathBuf> {
    std::fs::create_dir_all(output_dir)
        .map_err(|e| candle_core::Error::Msg(format!("mkdir failed: {}", e)))?;

    // First, get merged tensors by doing the same as save_pretrained_merged
    let device = candle_core::Device::Cpu;
    let mut tensors = load_base_tensors(model_dir, &device)?;
    let lora_tensors = get_lora_tensors(varmap);
    if !lora_tensors.is_empty() {
        let mut lora_pairs: HashMap<String, (Option<Tensor>, Option<Tensor>)> = HashMap::new();
        for (key, tensor) in &lora_tensors {
            let base_key = key.replace(".lora_a", "").replace(".lora_b", "");
            let entry = lora_pairs.entry(base_key).or_insert((None, None));
            if key.ends_with(".lora_a") {
                entry.0 = Some(tensor.clone());
            } else if key.ends_with(".lora_b") {
                entry.1 = Some(tensor.clone());
            }
        }
        for (base_key, (lora_a, lora_b)) in &lora_pairs {
            if let (Some(a), Some(b)) = (lora_a, lora_b) {
                let weight_key = format!("{}.weight", base_key);
                if let Some(base_weight) = tensors.get(&weight_key) {
                    let merged = merge_lora_into_weight(base_weight, a, b, 1.0)?;
                    tensors.insert(weight_key, merged);
                }
            }
        }
    }

    // Quantize 2D weight tensors to NF4
    let mut quantized: HashMap<String, Tensor> = HashMap::new();
    for (name, tensor) in &tensors {
        if tensor.rank() == 2 && name.contains(".weight")
            && !name.contains("embed") && !name.contains("norm")
        {
            let t_f32 = tensor.to_dtype(DType::F32)?;
            let (packed, scales) = quantize_to_nf4(&t_f32, block_size)?;
            quantized.insert(format!("{}.packed", name), packed);
            quantized.insert(format!("{}.scales", name), scales);
            println!("  Quantized: {}", name);
        } else {
            quantized.insert(name.clone(), tensor.to_dtype(DType::F32).unwrap_or(tensor.clone()));
        }
    }

    let safetensors_path = output_dir.join("model_4bit.safetensors");
    let num_entries = quantized.len();
    candle_core::safetensors::save(&quantized, &safetensors_path)?;

    // Config with quantization metadata
    let mut config = config_json.clone();
    if let Some(obj) = config.as_object_mut() {
        obj.insert("quantization_config".to_string(), serde_json::json!({
            "quant_method": "nf4", "bits": 4, "block_size": block_size, "quant_type": "nf4"
        }));
    }
    let config_path = output_dir.join("config.json");
    std::fs::write(&config_path, serde_json::to_string_pretty(&config)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?)
        .map_err(|e| candle_core::Error::Msg(format!("write config: {}", e)))?;

    println!("✓ 4-bit model saved to {}", output_dir.display());
    println!("  - model_4bit.safetensors ({} entries)", num_entries);
    Ok(safetensors_path)
}

/// Save merged model as GGUF format.
/// First saves as HF safetensors, then calls llama.cpp's convert script.
pub fn save_to_gguf(
    model_dir: &Path,
    varmap: &VarMap,
    config_json: &serde_json::Value,
    output_dir: &Path,
    quantization_type: &str,
) -> Result<PathBuf> {
    // 1. Save as merged HF safetensors in temp subfolder
    let hf_dir = output_dir.join("_hf_temp");
    save_pretrained_merged(model_dir, varmap, config_json, &hf_dir)?;

    // 2. Call Python to do the GGUF conversion
    let gguf_path = output_dir.join(format!("model-{}.gguf", quantization_type));
    std::fs::create_dir_all(output_dir)
        .map_err(|e| candle_core::Error::Msg(format!("mkdir: {}", e)))?;

    let result = std::process::Command::new("python3")
        .args(["-c", &format!(r#"
import subprocess, sys, os
hf_dir = "{hf_dir}"
gguf_out = "{gguf_out}"
quant_type = "{quant_type}"
convert_scripts = [
    os.path.expanduser("~/.local/bin/convert_hf_to_gguf.py"),
    "/usr/local/bin/convert_hf_to_gguf.py",
    "convert_hf_to_gguf.py",
]
for script in convert_scripts:
    if os.path.exists(script):
        subprocess.check_call([sys.executable, script, hf_dir, "--outfile", gguf_out, "--outtype", quant_type])
        print(f"GGUF saved: {{gguf_out}}")
        sys.exit(0)
try:
    import llama_cpp
    pkg_dir = os.path.dirname(llama_cpp.__file__)
    for sub in ["scripts", ""]:
        script = os.path.join(pkg_dir, sub, "convert_hf_to_gguf.py")
        if os.path.exists(script):
            subprocess.check_call([sys.executable, script, hf_dir, "--outfile", gguf_out, "--outtype", quant_type])
            print(f"GGUF saved: {{gguf_out}}")
            sys.exit(0)
except ImportError:
    pass
print("WARNING: No GGUF converter found. Install llama-cpp-python or llama.cpp.")
print("Merged HF safetensors at:", hf_dir)
print(f"Convert manually: python convert_hf_to_gguf.py {{hf_dir}} --outfile {{gguf_out}} --outtype {{quant_type}}")
"#,
            hf_dir = hf_dir.display(),
            gguf_out = gguf_path.display(),
            quant_type = quantization_type,
        )])
        .output()
        .map_err(|e| candle_core::Error::Msg(format!("python3 failed: {}", e)))?;

    let stdout = String::from_utf8_lossy(&result.stdout);
    let stderr = String::from_utf8_lossy(&result.stderr);
    if !stdout.is_empty() { println!("{}", stdout); }
    if !stderr.is_empty() { eprintln!("{}", stderr); }

    if gguf_path.exists() {
        println!("✓ GGUF model saved to {}", gguf_path.display());
        let _ = std::fs::remove_dir_all(&hf_dir);
    } else {
        println!("⚠ GGUF converter not found. Merged HF model at: {}", hf_dir.display());
    }

    Ok(gguf_path)
}
