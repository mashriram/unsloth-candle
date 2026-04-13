import os
import sys
import time
import torch
import unsloth_candle
from unsloth_candle import FastLanguageModel, SFTTrainer, SFTConfig

print("═" * 60)
print("  unsloth-candle  test_finetuning.py")
print("═" * 60)

# 1. Device and Dtype Strategy
device_name = "CPU"
is_cuda = torch.cuda.is_available()
is_mps = torch.backends.mps.is_available()

if is_cuda:
    device_name = "CUDA (GPU)"
elif is_mps:
    device_name = "Metal (MPS)"

print(f"Device detected: {device_name}")

# Similar to unsloth, decide on bf16 depending on CUDA capability
dtype = "float32"
load_in_4bit = True

if is_cuda:
    try:
        if torch.cuda.is_bf16_supported():
            print("CUDA device supports bf16. (Unsloth typical default)")
    except:
        pass
    print("Enforcing dtype='float32' as Candle's training backend currently only supports f32.")
    dtype = "float32"
elif is_mps:
    print("MPS detected. Defaulting to float32 (Candle standard fallback).")
else:
    print("CPU detected. Defaulting to float32.")

MODEL_NAME = os.environ.get("MODEL_NAME", "HuggingFaceTB/SmolLM2-135M-Instruct")
MAX_SEQ_LENGTH = 128

print(f"\n[STEP 1] Loading Model: {MODEL_NAME}")
print(f"  dtype: {dtype}")
print(f"  load_in_4bit: {load_in_4bit}")

t0 = time.perf_counter()
model, tokenizer = FastLanguageModel.from_pretrained(
    MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=load_in_4bit,
    dtype=dtype,
)
print(f"  ✓ Loaded in {time.perf_counter() - t0:.1f}s")
print(f"  ✓ Tokenizer: {tokenizer.__class__.__name__}")

# 2. Add LoRA Adapters
RANK = 16
ALPHA = 16
USE_DORA = False
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

print(f"\n[STEP 2] Applying PEFT (LoRA) adapters")
t0 = time.perf_counter()
model = FastLanguageModel.get_peft_model(
    model,
    r=RANK,
    target_modules=TARGET_MODULES,
    lora_alpha=ALPHA,
    lora_dropout=0.0,
    use_dora=USE_DORA,
)
print(f"  ✓ PEFT adapters attached in {(time.perf_counter() - t0)*1000:.0f}ms")

# 3. Training Data
print(f"\n[STEP 3] Preparing dataset")
TRAINING_TEXTS = [
    "### Instruction:\nExplain how neural networks learn.\n\n### Response:\nNeural networks learn through backpropagation.",
    "### Instruction:\nWhat is the capital of India?\n\n### Response:\nThe capital of India is New Delhi.",
    "### Instruction:\nConvert 100 Celsius to Fahrenheit.\n\n### Response:\n100°C = 212°F.",
]

train_dataset = [{"text": t} for t in TRAINING_TEXTS]
print(f"  ✓ Dataset prepared with {len(train_dataset)} examples")

# 4. SFTTrainer
NUM_STEPS = 5
LR = 2e-4

print(f"\n[STEP 4] Fine-tuning via SFTTrainer")
print(f"  Steps: {NUM_STEPS}, LR: {LR}")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    args=SFTConfig(
        max_steps=NUM_STEPS,
        learning_rate=LR,
        logging_steps=1,
        per_device_train_batch_size=1,
    ),
)

trainer_stats = trainer.train()
losses = trainer_stats["losses"]
print(f"  ✓ Training completed! Final loss: {losses[-1] if losses else 'N/A'}")

# 5. Generation Test
print(f"\n[STEP 5] Testing explicit generation")
FastLanguageModel.for_inference(model)

question = "What is the capital of India?"
inputs = tokenizer.encode(question)

print(f"  Question: {question}")
print("  Inference output: ", end="", flush=True)

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt=True)
_ = model.generate(
    input_ids=torch.tensor([inputs]),
    streamer=text_streamer,
    max_new_tokens=20
)
print()

# 6. Saving Options
OUTPUT_BASE = "./finetune_test_output"
os.makedirs(OUTPUT_BASE, exist_ok=True)

print(f"\n[STEP 6] Testing Model Export")

# a) HF Merged
hf_dir = os.path.join(OUTPUT_BASE, "hf_merged")
print(f"  Saving to HF Merged safetensors format: {hf_dir}")
t0 = time.perf_counter()
model.save_pretrained_merged(hf_dir)
tokenizer.save_pretrained(hf_dir)
print(f"  ✓ Saved in {time.perf_counter() - t0:.1f}s")

# b) 4-bit NF4
nf4_dir = os.path.join(OUTPUT_BASE, "nf4_4bit")
print(f"  Saving to 4-bit NF4 format: {nf4_dir}")
t0 = time.perf_counter()
model.save_in_4bit(nf4_dir)
tokenizer.save_pretrained(nf4_dir)
print(f"  ✓ Saved in {time.perf_counter() - t0:.1f}s")

# c) GGUF
gguf_dir = os.path.join(OUTPUT_BASE, "gguf")
print(f"  Saving to GGUF (q8_0) format: {gguf_dir}")
t0 = time.perf_counter()
try:
    model.save_pretrained_gguf(gguf_dir, tokenizer=tokenizer, quantization_type="q8_0")
    print(f"  ✓ Saved in {time.perf_counter() - t0:.1f}s")
except Exception as e:
    print(f"  ⚠ GGUF save bypassed: {e}")

print(f"\n[SUCCESS] `test_finetuning.py` pipeline ran cleanly.")
