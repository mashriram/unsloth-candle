//! Core model tests: Llama forward pass, QLoRA, DoRA, KV-cache
//! Run: cargo test --test test_core_models -- --nocapture

use candle_core::{DType, Device, Result, Tensor, Var};
use candle_nn::{VarBuilder, VarMap};
use unsloth_candle::model::AdapterLayer;
use unsloth_candle::model::llama::{Llama, Config, Cache};

// ─── Hardware detection ───────────────────────────────────────────────────────

pub fn best_device() -> Device {
    #[cfg(feature = "cuda")]
    if let Ok(d) = Device::new_cuda(0) {
        eprintln!("[hw] CUDA GPU");
        return d;
    }
    #[cfg(feature = "metal")]
    if let Ok(d) = Device::new_metal(0) {
        eprintln!("[hw] Metal (Apple Silicon)");
        return d;
    }
    eprintln!("[hw] CPU");
    Device::Cpu
}

// ─── Builder helpers ──────────────────────────────────────────────────────────

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

fn build_llama_varmap(h: usize, inter: usize, v: usize, l: usize,
                      nh: usize, nkv: usize, dev: &Device) -> Result<VarMap> {
    let mut vm = VarMap::new();
    fw(&mut vm, "embed_tokens.weight", &[v, h], dev)?;
    let hd = h / nh;
    for i in 0..l {
        let p = format!("layers.{}", i);
        fw(&mut vm, &format!("{}.self_attn.q_proj.weight", p), &[nh * hd, h], dev)?;
        fw(&mut vm, &format!("{}.self_attn.k_proj.weight", p), &[nkv * hd, h], dev)?;
        fw(&mut vm, &format!("{}.self_attn.v_proj.weight", p), &[nkv * hd, h], dev)?;
        fw(&mut vm, &format!("{}.self_attn.o_proj.weight", p), &[h, nh * hd], dev)?;
        fw(&mut vm, &format!("{}.mlp.gate_proj.weight", p),    &[inter, h], dev)?;
        fw(&mut vm, &format!("{}.mlp.up_proj.weight", p),      &[inter, h], dev)?;
        fw(&mut vm, &format!("{}.mlp.down_proj.weight", p),    &[h, inter], dev)?;
        fo(&mut vm, &format!("{}.input_layernorm.weight", p), h, dev)?;
        fo(&mut vm, &format!("{}.post_attention_layernorm.weight", p), h, dev)?;
    }
    fo(&mut vm, "norm.weight", h, dev)?;
    fw(&mut vm, "lm_head.weight", &[v, h], dev)?;
    Ok(vm)
}

fn llama_cfg(h: usize, inter: usize, v: usize, l: usize,
             nh: usize, nkv: usize, load_4bit: bool) -> Config {
    Config {
        hidden_size: h, intermediate_size: inter, vocab_size: v,
        num_hidden_layers: l, num_attention_heads: nh, num_key_value_heads: nkv,
        rms_norm_eps: 1e-5, rope_theta: 10000.0,
        bos_token_id: None, eos_token_id: None,
        max_position_embeddings: 128, tie_word_embeddings: false,
        use_flash_attn: false, rope_scaling: None,
        use_gradient_checkpointing: false, load_in_4bit: load_4bit,
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[test]
fn test_llama_fp32_forward() -> Result<()> {
    let dev = best_device();
    let (h, inter, v, l, nh) = (64, 128, 200, 2, 4);
    let mut vm = build_llama_varmap(h, inter, v, l, nh, nh, &dev)?;
    let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
    let cfg = llama_cfg(h, inter, v, l, nh, nh, false);
    let mut model = Llama::load(vb, &cfg)?;
    let mut cache = Cache::new(false, l);
    let input = Tensor::zeros((1, 8), DType::U32, &dev)?;
    let out = model.forward(&input, 0, &mut cache)?;
    assert_eq!(out.dims(), &[1, 8, v], "Llama fp32 shape mismatch");
    eprintln!("[Llama fp32] OK {:?}", out.dims());
    Ok(())
}

#[test]
fn test_llama_gqa_forward() -> Result<()> {
    // GQA: 8 query heads, 2 KV heads
    let dev = best_device();
    let (h, inter, v, l, nh, nkv) = (64, 128, 200, 2, 8, 2);
    let mut vm = build_llama_varmap(h, inter, v, l, nh, nkv, &dev)?;
    let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
    let cfg = llama_cfg(h, inter, v, l, nh, nkv, false);
    let mut model = Llama::load(vb, &cfg)?;
    let mut cache = Cache::new(false, l);
    let input = Tensor::zeros((1, 6), DType::U32, &dev)?;
    let out = model.forward(&input, 0, &mut cache)?;
    assert_eq!(out.dims(), &[1, 6, v]);
    eprintln!("[Llama GQA 8q/2kv] OK {:?}", out.dims());
    Ok(())
}

#[test]
fn test_llama_lora() -> Result<()> {
    let dev = best_device();
    let (h, inter, v, l, nh) = (64, 128, 200, 2, 4);
    let mut vm = build_llama_varmap(h, inter, v, l, nh, nh, &dev)?;
    let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
    let cfg = llama_cfg(h, inter, v, l, nh, nh, false);
    let mut model = Llama::load(vb, &cfg)?;
    model.apply_lora(
        vec!["q_proj".to_string(), "v_proj".to_string()],
        8, 16.0, 0.0, false, &mut vm
    )?;
    assert!(matches!(model.layers[0].attn.q_proj, AdapterLayer::LoRA(_)));
    let mut cache = Cache::new(false, l);
    let input = Tensor::zeros((1, 4), DType::U32, &dev)?;
    let out = model.forward(&input, 0, &mut cache)?;
    assert_eq!(out.dims(), &[1, 4, v]);
    eprintln!("[Llama+LoRA] OK {:?}", out.dims());
    Ok(())
}

#[test]
fn test_llama_qlora() -> Result<()> {
    // 4-bit quantization always runs on CPU
    let dev = Device::Cpu;
    let (h, inter, v, l, nh) = (64, 128, 200, 1, 4);
    let mut vm = build_llama_varmap(h, inter, v, l, nh, nh, &dev)?;
    let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
    let cfg = llama_cfg(h, inter, v, l, nh, nh, true);
    let mut model = Llama::load(vb, &cfg)?;

    // Layers should be 4bit
    assert!(matches!(model.layers[0].attn.q_proj, AdapterLayer::Linear4bit(_)),
        "QLoRA: q_proj must be Linear4bit after load");

    // Wrap 4bit layers with LoRA
    model.apply_lora(vec!["q_proj".to_string(), "k_proj".to_string()],
        8, 16.0, 0.0, false, &mut vm)?;

    // Verify nesting
    match &model.layers[0].attn.q_proj {
        AdapterLayer::LoRA(lo) => {
            assert!(matches!(*lo.base, AdapterLayer::Linear4bit(_)),
                "QLoRA: LoRA must wrap Linear4bit");
        },
        _ => panic!("Should be LoRA"),
    }
    let mut cache = Cache::new(false, l);
    let input = Tensor::zeros((1, 3), DType::U32, &dev)?;
    let out = model.forward(&input, 0, &mut cache)?;
    assert_eq!(out.dims(), &[1, 3, v]);
    eprintln!("[QLoRA] LoRA(Linear4bit) forward OK {:?}", out.dims());
    Ok(())
}

#[test]
fn test_llama_dora() -> Result<()> {
    let dev = Device::Cpu;
    let (h, inter, v, l, nh) = (64, 128, 200, 1, 4);
    let mut vm = build_llama_varmap(h, inter, v, l, nh, nh, &dev)?;
    let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
    let cfg = llama_cfg(h, inter, v, l, nh, nh, false);
    let mut model = Llama::load(vb, &cfg)?;
    model.apply_lora(vec!["q_proj".to_string()], 4, 8.0, 0.0, true, &mut vm)?;
    assert!(matches!(model.layers[0].attn.q_proj, AdapterLayer::DoRA(_)),
        "DoRA: q_proj should be DoRA variant");
    let mut cache = Cache::new(false, l);
    let input = Tensor::zeros((1, 3), DType::U32, &dev)?;
    let out = model.forward(&input, 0, &mut cache)?;
    assert_eq!(out.dims(), &[1, 3, v]);
    eprintln!("[DoRA] forward OK {:?}", out.dims());
    Ok(())
}

#[test]
fn test_kv_cache_shapes() -> Result<()> {
    let dev = Device::Cpu;
    let (h, inter, v, l, nh) = (64, 128, 200, 2, 4);
    let vm = build_llama_varmap(h, inter, v, l, nh, nh, &dev)?;
    let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
    let cfg = llama_cfg(h, inter, v, l, nh, nh, false);
    let mut model = Llama::load(vb, &cfg)?;

    // KV cache forward: first 3 tokens, then 1 more
    let mut cache = Cache::new(true, l);
    let tok3 = Tensor::from_vec(vec![1u32, 2, 3], (1, 3), &dev)?;
    let out3 = model.forward(&tok3, 0, &mut cache)?;
    assert_eq!(out3.dims(), &[1, 3, v]);

    let tok1 = Tensor::from_vec(vec![4u32], (1, 1), &dev)?;
    let out1 = model.forward(&tok1, 3, &mut cache)?;
    assert_eq!(out1.dims(), &[1, 1, v]);
    eprintln!("[KV Cache] prefix(3) + next(1) = shapes OK");
    Ok(())
}

#[test]
fn test_batch_forward() -> Result<()> {
    let dev = best_device();
    let (h, inter, v, l, nh) = (64, 128, 200, 2, 4);
    let vm = build_llama_varmap(h, inter, v, l, nh, nh, &dev)?;
    let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
    let cfg = llama_cfg(h, inter, v, l, nh, nh, false);
    let mut model = Llama::load(vb, &cfg)?;
    let mut cache = Cache::new(false, l);
    for bs in [1usize, 2, 4] {
        let input = Tensor::zeros((bs, 8), DType::U32, &dev)?;
        let start = std::time::Instant::now();
        let out = model.forward(&input, 0, &mut cache)?;
        assert_eq!(out.dims(), &[bs, 8, v]);
        eprintln!("[Batch bs={}] {:.1}ms", bs, start.elapsed().as_secs_f64() * 1000.0);
    }
    Ok(())
}
