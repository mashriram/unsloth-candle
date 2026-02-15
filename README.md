# Unsloth-Candle

**Unsloth-Candle** is a high-performance, memory-efficient library for Large Language Model (LLM) fine-tuning and inference, built using [Candle](https://github.com/huggingface/candle) (Rust) and exposed via Python bindings. It aims to replicate the core capabilities of [Unsloth](https://github.com/unslothai/unsloth) with a focus on speed, portability, and explicit memory management.

## 🚀 Key Features

*   **Universal Model Support**: unified architecture supporting a wide range of state-of-the-art LLMs:
    *   **Llama-3 / Llama-2** (and derivatives like Granite)
    *   **Mistral** & **Mixtral** (Sparse MoE)
    *   **Qwen2** & **Qwen2.5** (including **Qwen2-MoE**)
    *   **Gemma 2** (with Logit Soft-capping, GeGLU)
    *   **Phi-3** (Su-scaled RoPE, Sliding Window)
    *   **Cohere Command R**
    *   **GPT-NeoX / Pythia**
    *   **Vision Models**: Llava 1.5/1.6, Qwen2-VL, Pixtral
*   **Efficient Quantization**: Native support for **4-bit NF4** data types for minimal memory usage during loading and inference.
*   **LoRA & DoRA**: Full implementation of **Low-Rank Adaptation (LoRA)** and **Weight-Decomposed Low-Rank Adaptation (DoRA)** for efficient fine-tuning.
*   **Production Ready**: Includes sliding window attention, RoPE scaling (Linear, Dynamic), and Flash Attention v2 integration.
*   **Python Bindings**: Seamless `pip` installable package to use Rust performance within Python workflows.

## 🛠️ Installation

### Prerequisites
*   **Rust**: Install via `rustup` (`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`)
*   **Python**: 3.8+
*   **CUDA** (Optional but recommended for GPU acceleration)

### Build from Source

```bash
# Clone the repository
git clone https://github.com/your-username/unsloth-candle.git
cd unsloth-candle

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install build tools
pip install maturin

# Build and install the Python extension
maturin develop --release
```

## 💻 Usage

### Python API

The API mirrors the familiar `FastLanguageModel` interface.

```python
from unsloth_candle import FastLanguageModel

# 1. Load a model (supports 4-bit loading)
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/llama-3-8b-bnb-4bit",
    load_in_4bit=True,
    max_seq_length=2048
)

# 2. Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0,
    use_gradient_checkpointing=True,
    use_dora=False, # Set True for DoRA
)

# 3. Inference
inputs = tokenizer(["unsloth-candle is fast because"], return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(outputs[0]))

# 4. Training (Skeleton)
from trl import SFTTrainer
# ... Standard HuggingFace Trainer workflow works with the wrapped model ...
```

### Rust API

You can also use the library directly in Rust applications:

```rust
use unsloth_candle::model::{RustModel, Config};
use unsloth_candle::core::loader::load_4bit_model;

fn main() -> anyhow::Result<()> {
    // Load model weights and config
    let (mut model, config) = load_4bit_model("unsloth/llama-3-8b-bnb-4bit")?;
    
    // Apply LoRA
    model.apply_lora(vec!["q_proj".to_string()], 16, 32.0, 0.0, false)?;
    
    // Forward pass
    let logits = model.forward(&input_tensor, None, 0)?;
    
    Ok(())
}
```

## 🏗️ Architecture

The codebase is organized for modularity and performance:

*   `src/model/`: Implementations of individual architectures (Llama, Qwen2, etc.).
*   `src/layers/`: Common reusable layers (RotaryEmbedding, LoRALinear, MLP).
*   `src/core/`: Loading logic (`loader.rs`), Optimization (`optimizer.rs`), and Quantization utilities.
*   `src/kernels/`: Custom CUDA/CPU kernels for specific operations (if applicable).

## ✅ Supported Models Verification

| Architecture | LoRA | DoRA | 4-bit | Flash Attn |
| :--- | :---: | :---: | :---: | :---: |
| Llama 2/3 | ✅ | ✅ | ✅ | ✅ |
| Mistral / Mixtral | ✅ | ✅ | ✅ | ✅ |
| Qwen 2 / 2.5 | ✅ | ✅ | ✅ | ✅ |
| Gemma 2 | ✅ | ✅ | ✅ | ⚠️ (Softcap) |
| Phi-3 | ✅ | ✅ | ✅ | ✅ |
| Vision (Llava) | ✅ | ✅ | ✅ | N/A |

## 📜 License

MIT
