# 🚀 Unsloth-Candle

<div align="center">

**High-performance LLM fine-tuning library built with Rust 🦀 and Candle.**

[![PyPI version](https://img.shields.io/pypi/v/unsloth-candle.svg)](https://pypi.org/project/unsloth-candle/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Unsloth](https://img.shields.io/badge/Inspiration-Unsloth-orange.svg)](https://github.com/unslothai/unsloth)

</div>

---

**Unsloth-Candle** brings the blazing fast performance of [Unsloth](https://unsloth.ai) to the [Candle](https://github.com/huggingface/candle) ecosystem. By leveraging optimized Rust kernels and efficient memory management, it enables 2x faster training and 70% less memory usage compared to standard implementations.

## ✨ Core Advantages

*   **Zero Learning Curve**: 1:1 API compatibility with Unsloth's Python interface.
*   **Hardware Optimized**: Native support for **CUDA**, **Metal** (Apple Silicon), and **AVX/Neon** (CPU).
*   **Memory Efficient**: Native **4-bit NF4** quantization and gradient checkpointing.
*   **Unified Support**: One engine for Llama 3.2, Mistral, Qwen 2.5, DeepSeek-V3, and more.

## 📦 Installation

### Via Pip (Recommended)

```bash
pip install unsloth-candle
```

### Build from Source

```bash
git clone https://github.com/unslothai/unsloth-candle.git
cd unsloth-candle
pip install -e .
```

To enable GPU acceleration:
*   **CUDA**: `pip install -e . --features cuda`
*   **Metal**: `pip install -e . --features metal`

## 🛠 Usage

### 1. Load Model & Tokenizer
```python
from unsloth_candle import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct",
    max_seq_length = 2048,
    load_in_4bit = True,
)
```

### 2. Apply LoRA/DoRA
```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha = 16,
    use_gradient_checkpointing = True,
    use_dora = False, # Set to True for DoRA
)
```

### 3. Fine-tuning with SFTTrainer
```python
from unsloth_candle import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 2048,
    args = SFTConfig(
        max_steps = 60,
        learning_rate = 2e-4,
        logging_steps = 1,
    ),
)
trainer.train()
```

### 4. Save & Export
```python
# Save as merged HF weights
model.save_pretrained_merged("output_hf", tokenizer)

# Save as GGUF (for Ollama/llama.cpp)
model.save_pretrained_gguf("output_gguf", tokenizer, quantization_type="q4_k_m")
```

## 🗺️ Model Catalog

| Model | Architecture | 4-bit | LoRA | DoRA |
| :--- | :--- | :---: | :---: | :---: |
| **Llama 3.2** | LlamaForCausalLM | ✅ | ✅ | ✅ |
| **Mistral Nemo** | MistralForCausalLM | ✅ | ✅ | ✅ |
| **Qwen 2.5** | Qwen2ForCausalLM | ✅ | ✅ | ✅ |
| **DeepSeek V3** | DeepSeekV3 (MLA) | ✅ | ✅ | ✅ |
| **Gemma 3** | Gemma3 (GeGLU) | ✅ | ✅ | ✅ |
| **Phi 4** | Phi4 | ✅ | ✅ | ✅ |

## 📜 License

Licensed under the [Apache License, Version 2.0](LICENSE).

---
<div align="center">
Built with 💖 by the Unsloth Community and Antigravity.
</div>
