#!/usr/bin/env python3
"""
unsloth-candle  —  Real End-to-End Fine-tuning Pipeline
=======================================================

This script imports the actual compiled Rust library, loads a real model
from HuggingFace, applies LoRA/DoRA, runs training steps, and saves
in all 3 formats (HF merged, 4-bit NF4, GGUF).

Prerequisites:
    pip install maturin huggingface_hub transformers
    maturin develop                    # CPU
    maturin develop --features cuda    # NVIDIA GPU
    maturin develop --features metal   # Apple Silicon

Usage:
    python finetune_demo.py
"""

import os
import sys
import time

# ─── 1. Import unsloth_candle (real Rust library) ─────────────────────────────

import unsloth_candle
from unsloth_candle import FastLanguageModel, Trainer

print("═" * 60)
print("  unsloth-candle  Real E2E Fine-tuning Pipeline")
print("═" * 60)
print(f"  Python: {sys.version.split()[0]}")
print(f"  Lib:    unsloth_candle (Rust backend)")

# ─── 2. Load a real model from HuggingFace ────────────────────────────────────

MODEL_NAME = os.environ.get("MODEL_NAME", "unsloth/Llama-3.2-1B-Instruct")
LOAD_4BIT = os.environ.get("LOAD_4BIT", "0") == "1"
MAX_SEQ_LENGTH = int(os.environ.get("MAX_SEQ_LENGTH", "512"))

print(f"\n── STEP 1: Loading model ──")
print(f"  Model: {MODEL_NAME}")
print(f"  4-bit: {LOAD_4BIT}")

t0 = time.perf_counter()
model, tokenizer = FastLanguageModel.from_pretrained(
    MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=LOAD_4BIT,
    use_gradient_checkpointing=False,
)
print(f"  ✓ Loaded in {time.perf_counter() - t0:.1f}s")
print(f"  ✓ Tokenizer loaded ({tokenizer.__class__.__name__})")

# We use the tokenizer returned by the model
HAS_TOKENIZER = True

# ─── 3. Apply LoRA adapters ──────────────────────────────────────────────────

RANK = int(os.environ.get("LORA_RANK", "16"))
ALPHA = float(os.environ.get("LORA_ALPHA", "16"))
USE_DORA = os.environ.get("USE_DORA", "0") == "1"
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

print(f"\n── STEP 2: Applying {'DoRA' if USE_DORA else 'LoRA'} adapters ──")
print(f"  Rank:    {RANK}")
print(f"  Alpha:   {ALPHA}")
print(f"  Targets: {TARGET_MODULES}")

t0 = time.perf_counter()
model.apply_lora(
    target_modules=TARGET_MODULES,
    rank=RANK,
    alpha=ALPHA,
    dropout=0.0,
    use_dora=USE_DORA,
)
print(f"  ✓ Adapters applied in {(time.perf_counter() - t0)*1000:.0f}ms")

# ─── 4. Tokenize training data ───────────────────────────────────────────────

print(f"\n── STEP 3: Preparing training data ──")

TRAINING_TEXTS = [
    "### Instruction:\nExplain how neural networks learn.\n\n### Response:\nNeural networks learn through backpropagation. The algorithm computes gradients of a loss function with respect to the weights.",
    "### Instruction:\nWrite a Python function to compute factorial.\n\n### Response:\ndef factorial(n):\n    if n <= 1: return 1\n    return n * factorial(n - 1)",
    "### Instruction:\nWhat is the capital of India?\n\n### Response:\nThe capital of India is New Delhi.",
    "### Instruction:\nExplain transformers in one sentence.\n\n### Response:\nTransformers use self-attention mechanisms to process all positions in a sequence simultaneously, enabling efficient parallel training.",
    "### Instruction:\nConvert 100 Celsius to Fahrenheit.\n\n### Response:\n100°C = 212°F. The formula is F = C × 9/5 + 32.",
]


def tokenize(text: str) -> list:
    return tokenizer.encode(text, truncation=True, max_length=MAX_SEQ_LENGTH)


train_ids = [tokenize(t) for t in TRAINING_TEXTS]
avg_len = sum(len(ids) for ids in train_ids) / len(train_ids)
print(f"  Samples: {len(train_ids)}")
print(f"  Avg tokens: {avg_len:.0f}")

# ─── 5. Training loop ────────────────────────────────────────────────────────

NUM_STEPS = int(os.environ.get("NUM_STEPS", "10"))
LR = float(os.environ.get("LEARNING_RATE", "2e-4"))

print(f"\n── STEP 4: Fine-tuning ──")
print(f"  Steps: {NUM_STEPS}")
print(f"  LR:    {LR}")

trainer = Trainer(model)
trainer.configure_optimizer(LR)

losses = []
for step in range(NUM_STEPS):
    sample = train_ids[step % len(train_ids)]
    t0 = time.perf_counter()
    loss = trainer.train_step(sample)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    losses.append(loss)
    tok_per_sec = len(sample) / (elapsed_ms / 1000) if elapsed_ms > 0 else 0
    print(f"  step {step+1:3d}/{NUM_STEPS}  loss={loss:.4f}  "
          f"{elapsed_ms:.0f}ms  {tok_per_sec:.0f} tok/s")

if len(losses) >= 2:
    trend = "↓ (improving)" if losses[-1] < losses[0] else "→ (flat/diverging)"
    print(f"\n  Loss: {losses[0]:.4f} → {losses[-1]:.4f} {trend}")

# ─── 6. Forward pass test ────────────────────────────────────────────────────

print(f"\n── STEP 5: Forward pass verification ──")
test_input = tokenize("Hello, how are you?")
result = model.forward(test_input)
print(f"  ✓ {result}")

# ─── 7. Save: HF merged safetensors ──────────────────────────────────────────

OUTPUT_BASE = os.environ.get("OUTPUT_DIR", "./output")

print(f"\n── STEP 6a: Save as HF merged safetensors ──")
hf_dir = os.path.join(OUTPUT_BASE, "hf_merged")
t0 = time.perf_counter()
path = model.save_pretrained_merged(hf_dir)
print(f"  Saved: {path} ({time.perf_counter() - t0:.1f}s)")

# Copy tokenizer files if available
if HAS_TOKENIZER:
    try:
        tokenizer.save_pretrained(hf_dir)
        print(f"  ✓ Tokenizer saved to {hf_dir}")
    except Exception as e:
        print(f"  ⚠ Tokenizer save failed: {e}")

# ─── 8. Save: 4-bit NF4 ──────────────────────────────────────────────────────

print(f"\n── STEP 6b: Save in 4-bit NF4 format ──")
nf4_dir = os.path.join(OUTPUT_BASE, "nf4_4bit")
t0 = time.perf_counter()
path = model.save_in_4bit(nf4_dir, block_size=64)
print(f"  Saved: {path} ({time.perf_counter() - t0:.1f}s)")

# ─── 9. Save: GGUF ───────────────────────────────────────────────────────────

print(f"\n── STEP 6c: Save as GGUF ──")
gguf_dir = os.path.join(OUTPUT_BASE, "gguf")
t0 = time.perf_counter()
try:
    path = model.save_to_gguf(gguf_dir, quantization_type="q8_0")
    print(f"  Saved: {path} ({time.perf_counter() - t0:.1f}s)")
except Exception as e:
    print(f"  ⚠ GGUF save: {e}")
    print("  (Install llama-cpp-python or llama.cpp for GGUF support)")

# ─── 10. Summary ─────────────────────────────────────────────────────────────

print(f"\n{'═' * 60}")
print(f"  ✅ Pipeline complete!")
print(f"{'═' * 60}")
print(f"  Model:       {MODEL_NAME}")
print(f"  Adapter:     {'DoRA' if USE_DORA else 'LoRA'} rank={RANK}")
print(f"  Steps:       {NUM_STEPS}")
print(f"  Final loss:  {losses[-1]:.4f}")
print(f"  Outputs:")
print(f"    HF merged: {os.path.abspath(hf_dir)}")
print(f"    4-bit NF4: {os.path.abspath(nf4_dir)}")
print(f"    GGUF:      {os.path.abspath(gguf_dir)}")
print()
print("To push to HuggingFace Hub:")
print(f"  huggingface-cli upload your-username/model-name {os.path.abspath(hf_dir)}")
print()
print("To run with Ollama (after GGUF save):")
print(f"  ollama create mymodel -f {os.path.abspath(gguf_dir)}/Modelfile")
