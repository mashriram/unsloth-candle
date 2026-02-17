use candle_core::{Device, DType, Tensor, Result, Var};
use candle_nn::{VarBuilder, VarMap};
use unsloth_candle::model::{Llama, LlamaConfig, AdapterLayer};
use unsloth_candle::model::linear4bit::Linear4bit;

#[test]
fn test_qlora_loading() -> Result<()> {
    eprintln!("DEBUG: Starting test_qlora_loading");
    
    // 1. Create a dummy Llama config with load_in_4bit = true
    let mut config = LlamaConfig {
        hidden_size: 64,
        intermediate_size: 128,
        vocab_size: 100,
        num_hidden_layers: 1,
        num_attention_heads: 4,
        num_key_value_heads: 4,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        bos_token_id: None,
        eos_token_id: None,
        max_position_embeddings: 128,
        tie_word_embeddings: false,
        use_flash_attn: false,
        rope_scaling: None,
        use_gradient_checkpointing: false,
        load_in_4bit: true,
    };

    let device = Device::Cpu;
    let dtype = DType::F32;
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, dtype, &device);
    eprintln!("DEBUG: VarMap created");

use candle_core::{Var}; // Needed for Var

// ...

    fn fill_linear(varmap: &mut VarMap, path: &str, shape: (usize, usize)) -> Result<()> {
        let t = Tensor::randn(0f32, 1f32, shape, &Device::Cpu)?;
        let var = Var::from_tensor(&t)?;
        varmap.data().lock().unwrap().insert(path.to_string(), var);
        eprintln!("DEBUG: Filled {}", path);
        Ok(())
    }

    fill_linear(&mut varmap, "embed_tokens.weight", (100, 64))?;
    fill_linear(&mut varmap, "layers.0.self_attn.q_proj.weight", (64, 64))?;
    fill_linear(&mut varmap, "layers.0.self_attn.k_proj.weight", (64, 64))?;
    fill_linear(&mut varmap, "layers.0.self_attn.v_proj.weight", (64, 64))?;
    fill_linear(&mut varmap, "layers.0.self_attn.o_proj.weight", (64, 64))?;
    fill_linear(&mut varmap, "layers.0.mlp.gate_proj.weight", (128, 64))?;
    fill_linear(&mut varmap, "layers.0.mlp.up_proj.weight", (128, 64))?;
    fill_linear(&mut varmap, "layers.0.mlp.down_proj.weight", (64, 128))?;
    // Norms
    {
        let t = Tensor::ones((64,), DType::F32, &Device::Cpu)?;
        varmap.data().lock().unwrap().insert("layers.0.input_layernorm.weight".to_string(), Var::from_tensor(&t)?);
        eprintln!("DEBUG: Filled layers.0.input_layernorm.weight");
        
        let t = Tensor::ones((64,), DType::F32, &Device::Cpu)?;
        varmap.data().lock().unwrap().insert("layers.0.post_attention_layernorm.weight".to_string(), Var::from_tensor(&t)?);
        eprintln!("DEBUG: Filled layers.0.post_attention_layernorm.weight");

        let t = Tensor::ones((64,), DType::F32, &Device::Cpu)?;
        varmap.data().lock().unwrap().insert("norm.weight".to_string(), Var::from_tensor(&t)?); 
        eprintln!("DEBUG: Filled norm.weight");
    }
    fill_linear(&mut varmap, "lm_head.weight", (100, 64))?;

    eprintln!("DEBUG: All weights filled");
    eprintln!("VarMap keys: {:?}", varmap.data().lock().unwrap().keys());
    
    // Debug asserting existence
    {
        let data = varmap.data().lock().unwrap();
        if !data.contains_key("embed_tokens.weight") {
             panic!("varmap missing embed_tokens.weight!");
        }
    }

    // Debug: Try to get embedding weight manually
    eprintln!("DEBUG: Attempting manual vb.get");
    match vb.pp("embed_tokens").get((100, 64), "weight") {
        Ok(t) => eprintln!("Found embed_tokens.weight: {:?}", t),
        Err(e) => eprintln!("Failed to find embed_tokens.weight via vb: {}", e),
    }

    // 3. Load Model
    eprintln!("DEBUG: Calling Llama::load...");
    // Pass vb without model prefix as we populated without it
    let mut model = Llama::load(vb, &config)?;
    eprintln!("DEBUG: Llama::load success");

    // 4. Verify Layers are Linear4bit
    let block = &model.layers[0];
    if let AdapterLayer::Linear4bit(_) = &block.attn.q_proj {
        println!("q_proj is Linear4bit - Success");
    } else {
        panic!("q_proj should be Linear4bit");
    }

    // 5. Apply LoRA (QLoRA)
    let target_modules = vec!["q_proj".to_string()];
    eprintln!("DEBUG: Applying LoRA...");
    model.apply_lora(target_modules, 8, 16.0, 0.0, false, &mut varmap)?;
    eprintln!("DEBUG: LoRA applied");

    // 6. Verify Layer is now LoRA wrapped around Linear4bit?
    let block = &model.layers[0];
    match &block.attn.q_proj {
        AdapterLayer::LoRA(l) => {
            if let AdapterLayer::Linear4bit(_) = &*l.base {
                println!("q_proj is LoRA(Linear4bit) - Success");
            } else {
                panic!("Base layer of LoRA should be Linear4bit");
            }
        },
        _ => panic!("q_proj should be LoRA"),
    }

    Ok(())
}
