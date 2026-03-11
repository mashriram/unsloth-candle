//! New architecture tests: Mistral, Qwen3, Gemma3, Granite, OLMo, StarCoder2, Phi4
//! Run: cargo test --test test_new_models -- --nocapture

use candle_core::{DType, Device, Result, Tensor, Var};
use candle_nn::{VarBuilder, VarMap};

fn best_device() -> Device {
    #[cfg(feature = "cuda")]
    if let Ok(d) = Device::new_cuda(0) { eprintln!("[hw] CUDA"); return d; }
    #[cfg(feature = "metal")]
    if let Ok(d) = Device::new_metal(0) { eprintln!("[hw] Metal"); return d; }
    eprintln!("[hw] CPU"); Device::Cpu
}

fn fw(vm: &mut VarMap, path: &str, shape: &[usize], dev: &Device) -> Result<()> {
    let t = Tensor::randn(0f32, 0.02f32, shape, dev)?;
    vm.data().lock().unwrap().insert(path.to_string(), Var::from_tensor(&t)?);
    Ok(())
}
fn fo(vm: &mut VarMap, path: &str, n: usize, dev: &Device) -> Result<()> {
    let t = Tensor::ones((n,), DType::F32, dev)?;
    vm.data().lock().unwrap().insert(path.to_string(), Var::from_tensor(&t)?);
    Ok(())
}
fn fz(vm: &mut VarMap, path: &str, n: usize, dev: &Device) -> Result<()> {
    let t = Tensor::zeros((n,), DType::F32, dev)?;
    vm.data().lock().unwrap().insert(path.to_string(), Var::from_tensor(&t)?);
    Ok(())
}

// ─── Mistral ─────────────────────────────────────────────────────────────────

#[test]
fn test_mistral_sliding_window() -> Result<()> {
    let dev = best_device();
    let (h, inter, v, nl, nh, nkv) = (64, 128, 200, 2, 4, 2);
    let hd = h / nh;
    let mut vm = VarMap::new();
    fw(&mut vm, "model.embed_tokens.weight", &[v, h], &dev)?;
    for i in 0..nl {
        let p = format!("model.layers.{}", i);
        fw(&mut vm, &format!("{}.self_attn.q_proj.weight", p), &[nh*hd, h], &dev)?;
        fw(&mut vm, &format!("{}.self_attn.k_proj.weight", p), &[nkv*hd, h], &dev)?;
        fw(&mut vm, &format!("{}.self_attn.v_proj.weight", p), &[nkv*hd, h], &dev)?;
        fw(&mut vm, &format!("{}.self_attn.o_proj.weight", p), &[h, nh*hd], &dev)?;
        fw(&mut vm, &format!("{}.mlp.gate_proj.weight", p), &[inter, h], &dev)?;
        fw(&mut vm, &format!("{}.mlp.up_proj.weight", p), &[inter, h], &dev)?;
        fw(&mut vm, &format!("{}.mlp.down_proj.weight", p), &[h, inter], &dev)?;
        fo(&mut vm, &format!("{}.input_layernorm.weight", p), h, &dev)?;
        fo(&mut vm, &format!("{}.post_attention_layernorm.weight", p), h, &dev)?;
    }
    fo(&mut vm, "model.norm.weight", h, &dev)?;
    fw(&mut vm, "lm_head.weight", &[v, h], &dev)?;

    let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
    use unsloth_candle::model::mistral::{Mistral, Config, MistralModel};
    let cfg = Config {
        hidden_size: h, intermediate_size: inter, vocab_size: v,
        num_hidden_layers: nl, num_attention_heads: nh, num_key_value_heads: nkv,
        rms_norm_eps: 1e-5, rope_theta: 10000.0, bos_token_id: None, eos_token_id: None,
        max_position_embeddings: 128, tie_word_embeddings: false, use_flash_attn: false,
        sliding_window: Some(16),
    };
    let model = Mistral::load(vb, &cfg)?;
    let mut w = MistralModel::new(model, cfg, dev.clone(), DType::F32, vm);
    let input = Tensor::zeros((1, 4), DType::U32, &dev)?;
    let out = w.forward(&input, 0)?;
    assert_eq!(out.dims(), &[1, 4, v]);
    eprintln!("[Mistral SWA=16] OK {:?}", out.dims());
    Ok(())
}

// ─── Qwen3 ───────────────────────────────────────────────────────────────────

#[test]
fn test_qwen3_qknorm() -> Result<()> {
    let dev = best_device();
    let (h, inter, v, nl, nh, nkv, hd) = (64, 128, 200, 2, 4, 2, 16);
    let mut vm = VarMap::new();
    fw(&mut vm, "model.embed_tokens.weight", &[v, h], &dev)?;
    for i in 0..nl {
        let p = format!("model.layers.{}", i);
        fw(&mut vm, &format!("{}.self_attn.q_proj.weight", p), &[nh*hd, h], &dev)?;
        fw(&mut vm, &format!("{}.self_attn.k_proj.weight", p), &[nkv*hd, h], &dev)?;
        fw(&mut vm, &format!("{}.self_attn.v_proj.weight", p), &[nkv*hd, h], &dev)?;
        fw(&mut vm, &format!("{}.self_attn.o_proj.weight", p), &[h, nh*hd], &dev)?;
        fo(&mut vm, &format!("{}.self_attn.q_norm.weight", p), hd, &dev)?;
        fo(&mut vm, &format!("{}.self_attn.k_norm.weight", p), hd, &dev)?;
        fw(&mut vm, &format!("{}.mlp.gate_proj.weight", p), &[inter, h], &dev)?;
        fw(&mut vm, &format!("{}.mlp.up_proj.weight", p), &[inter, h], &dev)?;
        fw(&mut vm, &format!("{}.mlp.down_proj.weight", p), &[h, inter], &dev)?;
        fo(&mut vm, &format!("{}.input_layernorm.weight", p), h, &dev)?;
        fo(&mut vm, &format!("{}.post_attention_layernorm.weight", p), h, &dev)?;
    }
    fo(&mut vm, "model.norm.weight", h, &dev)?;
    fw(&mut vm, "lm_head.weight", &[v, h], &dev)?;

    let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
    use unsloth_candle::model::qwen3::{Qwen3, Config, Qwen3Model};
    let cfg = Config {
        hidden_size: h, intermediate_size: inter, vocab_size: v,
        num_hidden_layers: nl, num_attention_heads: nh, num_key_value_heads: nkv,
        rms_norm_eps: 1e-5, rope_theta: 10000.0, bos_token_id: None, eos_token_id: None,
        max_position_embeddings: 128, tie_word_embeddings: false, head_dim: hd,
    };
    let model = Qwen3::load(vb, &cfg)?;
    let mut w = Qwen3Model::new(model, cfg, dev.clone(), DType::F32, vm);
    let input = Tensor::zeros((1, 4), DType::U32, &dev)?;
    let out = w.forward(&input, 0)?;
    assert_eq!(out.dims(), &[1, 4, v]);
    eprintln!("[Qwen3 QK-Norm] OK {:?}", out.dims());

    // LoRA after QK-Norm
    w.apply_lora(vec!["self_attn.q_proj".to_string(), "self_attn.v_proj".to_string()],
        8, 16.0, 0.0, false)?;
    let out2 = w.forward(&input, 0)?;
    assert_eq!(out2.dims(), &[1, 4, v]);
    eprintln!("[Qwen3+LoRA] OK");
    Ok(())
}

// ─── Gemma3 ──────────────────────────────────────────────────────────────────

#[test]
fn test_gemma3_interleaved_attention() -> Result<()> {
    let dev = best_device();
    let (h, inter, v, nl, nh, nkv, hd) = (64, 128, 200, 3, 4, 2, 16);
    let mut vm = VarMap::new();
    fw(&mut vm, "model.embed_tokens.weight", &[v, h], &dev)?;
    for i in 0..nl {
        let p = format!("model.layers.{}", i);
        fw(&mut vm, &format!("{}.self_attn.q_proj.weight", p), &[nh*hd, h], &dev)?;
        fw(&mut vm, &format!("{}.self_attn.k_proj.weight", p), &[nkv*hd, h], &dev)?;
        fw(&mut vm, &format!("{}.self_attn.v_proj.weight", p), &[nkv*hd, h], &dev)?;
        fw(&mut vm, &format!("{}.self_attn.o_proj.weight", p), &[h, nh*hd], &dev)?;
        fo(&mut vm, &format!("{}.self_attn.q_norm.weight", p), hd, &dev)?;
        fo(&mut vm, &format!("{}.self_attn.k_norm.weight", p), hd, &dev)?;
        fw(&mut vm, &format!("{}.mlp.gate_proj.weight", p), &[inter, h], &dev)?;
        fw(&mut vm, &format!("{}.mlp.up_proj.weight", p), &[inter, h], &dev)?;
        fw(&mut vm, &format!("{}.mlp.down_proj.weight", p), &[h, inter], &dev)?;
        fo(&mut vm, &format!("{}.input_layernorm.weight", p), h, &dev)?;
        fo(&mut vm, &format!("{}.post_attention_layernorm.weight", p), h, &dev)?;
        fo(&mut vm, &format!("{}.pre_feedforward_layernorm.weight", p), h, &dev)?;
        fo(&mut vm, &format!("{}.post_feedforward_layernorm.weight", p), h, &dev)?;
    }
    fo(&mut vm, "model.norm.weight", h, &dev)?;
    fw(&mut vm, "lm_head.weight", &[v, h], &dev)?;

    let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
    use unsloth_candle::model::gemma3::{Gemma3, Config, Gemma3Model};
    let cfg = Config {
        hidden_size: h, intermediate_size: inter, vocab_size: v,
        num_hidden_layers: nl, num_attention_heads: nh, num_key_value_heads: nkv,
        rms_norm_eps: 1e-5, rope_theta: 10000.0, bos_token_id: None, eos_token_id: None,
        max_position_embeddings: 128, head_dim: hd, sliding_window: Some(16),
        attn_logit_softcapping: None, final_logit_softcapping: None,
        tie_word_embeddings: false, sliding_window_pattern: 2,
    };
    let model = Gemma3::load(vb, &cfg)?;
    let mut w = Gemma3Model::new(model, cfg, dev.clone(), DType::F32, vm);
    let input = Tensor::zeros((1, 4), DType::U32, &dev)?;
    let out = w.forward(&input, 0)?;
    assert_eq!(out.dims(), &[1, 4, v]);
    eprintln!("[Gemma3 local+global attn] OK {:?}", out.dims());
    Ok(())
}

// ─── Granite ─────────────────────────────────────────────────────────────────

#[test]
fn test_granite_embedding_multiplier() -> Result<()> {
    let dev = best_device();
    let (h, inter, v, nl, nh) = (64, 128, 200, 2, 4);
    let mut vm = VarMap::new();
    fw(&mut vm, "model.embed_tokens.weight", &[v, h], &dev)?;
    for i in 0..nl {
        let p = format!("model.layers.{}", i);
        for proj in ["q_proj","k_proj","v_proj","o_proj"] {
            fw(&mut vm, &format!("{}.self_attn.{}.weight", p, proj), &[h, h], &dev)?;
        }
        fw(&mut vm, &format!("{}.mlp.gate_proj.weight", p), &[inter, h], &dev)?;
        fw(&mut vm, &format!("{}.mlp.up_proj.weight", p), &[inter, h], &dev)?;
        fw(&mut vm, &format!("{}.mlp.down_proj.weight", p), &[h, inter], &dev)?;
        fo(&mut vm, &format!("{}.input_layernorm.weight", p), h, &dev)?;
        fo(&mut vm, &format!("{}.post_attention_layernorm.weight", p), h, &dev)?;
    }
    fo(&mut vm, "model.norm.weight", h, &dev)?;
    fw(&mut vm, "lm_head.weight", &[v, h], &dev)?;

    let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
    use unsloth_candle::model::granite::{Granite, Config, GraniteModel};
    let cfg = Config {
        hidden_size: h, intermediate_size: inter, vocab_size: v,
        num_hidden_layers: nl, num_attention_heads: nh, num_key_value_heads: nh,
        rms_norm_eps: 1e-5, rope_theta: 10000.0, bos_token_id: None, eos_token_id: None,
        max_position_embeddings: 128, tie_word_embeddings: false,
        embedding_multiplier: 12.0, logits_scaling: 0.1, use_bias: false,
    };
    let model = Granite::load(vb, &cfg)?;
    let mut w = GraniteModel::new(model, cfg, dev.clone(), DType::F32, vm);
    let input = Tensor::zeros((1, 4), DType::U32, &dev)?;
    let out = w.forward(&input, 0)?;
    assert_eq!(out.dims(), &[1, 4, v]);
    eprintln!("[Granite emb_mult=12 logits_scale=0.1] OK {:?}", out.dims());
    Ok(())
}

// ─── OLMo ────────────────────────────────────────────────────────────────────

#[test]
fn test_olmo_layernorm_and_qkv_clip() -> Result<()> {
    let dev = best_device();
    let (h, inter, v, nl, nh) = (64, 128, 200, 2, 4);
    let mut vm = VarMap::new();
    fw(&mut vm, "model.embed_tokens.weight", &[v, h], &dev)?;
    for i in 0..nl {
        let p = format!("model.layers.{}", i);
        for proj in ["q_proj","k_proj","v_proj","o_proj"] {
            fw(&mut vm, &format!("{}.self_attn.{}.weight", p, proj), &[h, h], &dev)?;
        }
        fw(&mut vm, &format!("{}.mlp.gate_proj.weight", p), &[inter, h], &dev)?;
        fw(&mut vm, &format!("{}.mlp.up_proj.weight", p), &[inter, h], &dev)?;
        fw(&mut vm, &format!("{}.mlp.down_proj.weight", p), &[h, inter], &dev)?;
        fo(&mut vm, &format!("{}.input_layernorm.weight", p), h, &dev)?;
        fz(&mut vm, &format!("{}.input_layernorm.bias", p), h, &dev)?;
        fo(&mut vm, &format!("{}.post_attention_layernorm.weight", p), h, &dev)?;
        fz(&mut vm, &format!("{}.post_attention_layernorm.bias", p), h, &dev)?;
    }
    fo(&mut vm, "model.norm.weight", h, &dev)?;
    fz(&mut vm, "model.norm.bias", h, &dev)?;
    fw(&mut vm, "lm_head.weight", &[v, h], &dev)?;

    let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
    use unsloth_candle::model::olmo::{Olmo, Config, OlmoModel};
    let cfg = Config {
        hidden_size: h, intermediate_size: inter, vocab_size: v,
        num_hidden_layers: nl, num_attention_heads: nh, num_key_value_heads: nh,
        layer_norm_eps: 1e-5, rope_theta: 10000.0, bos_token_id: None, eos_token_id: None,
        max_position_embeddings: 128, clip_qkv: Some(8.0), tie_word_embeddings: false,
    };
    let model = Olmo::load(vb, &cfg)?;
    let mut w = OlmoModel::new(model, cfg, dev.clone(), DType::F32, vm);
    let input = Tensor::zeros((1, 4), DType::U32, &dev)?;
    let out = w.forward(&input, 0)?;
    assert_eq!(out.dims(), &[1, 4, v]);
    eprintln!("[OLMo LayerNorm clip_qkv=8] OK {:?}", out.dims());
    Ok(())
}

// ─── StarCoder2 ──────────────────────────────────────────────────────────────

#[test]
fn test_starcoder2_code_model() -> Result<()> {
    let dev = best_device();
    let (h, inter, v, nl, nh) = (64, 128, 200, 2, 4);
    let mut vm = VarMap::new();
    fw(&mut vm, "model.embed_tokens.weight", &[v, h], &dev)?;
    for i in 0..nl {
        let p = format!("model.layers.{}", i);
        for proj in ["q_proj","k_proj","v_proj","o_proj"] {
            fw(&mut vm, &format!("{}.attn.{}.weight", p, proj), &[h, h], &dev)?;
            fz(&mut vm, &format!("{}.attn.{}.bias", p, proj), h, &dev)?;
        }
        fw(&mut vm, &format!("{}.mlp.c_fc.weight", p), &[inter, h], &dev)?;
        fz(&mut vm, &format!("{}.mlp.c_fc.bias", p), inter, &dev)?;
        fw(&mut vm, &format!("{}.mlp.c_proj.weight", p), &[h, inter], &dev)?;
        fz(&mut vm, &format!("{}.mlp.c_proj.bias", p), h, &dev)?;
        fo(&mut vm, &format!("{}.ln_1.weight", p), h, &dev)?;
        fz(&mut vm, &format!("{}.ln_1.bias", p), h, &dev)?;
        fo(&mut vm, &format!("{}.ln_2.weight", p), h, &dev)?;
        fz(&mut vm, &format!("{}.ln_2.bias", p), h, &dev)?;
    }
    fo(&mut vm, "model.norm.weight", h, &dev)?;
    fz(&mut vm, "model.norm.bias", h, &dev)?;
    fw(&mut vm, "lm_head.weight", &[v, h], &dev)?;

    let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
    use unsloth_candle::model::starcoder2::{StarCoder2, Config, StarCoder2Model};
    let cfg = Config {
        hidden_size: h, intermediate_size: inter, vocab_size: v,
        num_hidden_layers: nl, num_attention_heads: nh, num_key_value_heads: nh,
        norm_epsilon: 1e-5, rope_theta: 10000.0, bos_token_id: None, eos_token_id: None,
        max_position_embeddings: 128, sliding_window: None, tie_word_embeddings: false, use_bias: true,
    };
    let model = StarCoder2::load(vb, &cfg)?;
    let mut w = StarCoder2Model::new(model, cfg, dev.clone(), DType::F32, vm);
    let input = Tensor::zeros((1, 4), DType::U32, &dev)?;
    let out = w.forward(&input, 0)?;
    assert_eq!(out.dims(), &[1, 4, v]);
    eprintln!("[StarCoder2 GELU+biased] OK {:?}", out.dims());
    Ok(())
}

// ─── Phi-4 ───────────────────────────────────────────────────────────────────

#[test]
fn test_phi4_fused_proj() -> Result<()> {
    let dev = best_device();
    let (h, inter, v, nl, nh, nkv, hd) = (64, 128, 200, 2, 4, 2, 16);
    let mut vm = VarMap::new();
    fw(&mut vm, "model.embed_tokens.weight", &[v, h], &dev)?;
    for i in 0..nl {
        let p = format!("model.layers.{}", i);
        let fused = nh * hd + 2 * nkv * hd;
        fw(&mut vm, &format!("{}.self_attn.qkv_proj.weight", p), &[fused, h], &dev)?;
        fw(&mut vm, &format!("{}.self_attn.o_proj.weight", p), &[h, nh*hd], &dev)?;
        fw(&mut vm, &format!("{}.mlp.gate_up_proj.weight", p), &[2*inter, h], &dev)?;
        fw(&mut vm, &format!("{}.mlp.down_proj.weight", p), &[h, inter], &dev)?;
        fo(&mut vm, &format!("{}.input_layernorm.weight", p), h, &dev)?;
        fo(&mut vm, &format!("{}.post_attention_layernorm.weight", p), h, &dev)?;
    }
    fo(&mut vm, "model.norm.weight", h, &dev)?;
    fw(&mut vm, "lm_head.weight", &[v, h], &dev)?;

    let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
    use unsloth_candle::model::phi4::{Phi4, Config, Phi4Model};
    let cfg = Config {
        hidden_size: h, intermediate_size: inter, vocab_size: v,
        num_hidden_layers: nl, num_attention_heads: nh, num_key_value_heads: nkv,
        rms_norm_eps: 1e-5, rope_theta: 10000.0, bos_token_id: None, eos_token_id: None,
        max_position_embeddings: 128, tie_word_embeddings: false, use_flash_attn: false, rope_scaling: None,
    };
    let model = Phi4::load(vb, &cfg)?;
    let mut w = Phi4Model::new(model, cfg, dev.clone(), DType::F32, vm);
    let input = Tensor::zeros((1, 4), DType::U32, &dev)?;
    let out = w.forward(&input, 0)?;
    assert_eq!(out.dims(), &[1, 4, v]);
    eprintln!("[Phi4 fused qkv+gate_up] OK {:?}", out.dims());
    Ok(())
}
