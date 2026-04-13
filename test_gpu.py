import subprocess
import torch
import time

def print_memory(label):
    mem = subprocess.getoutput("nvidia-smi --query-gpu=memory.used --format=csv,noheader")
    print(f"[{label}] GPU Memory Used: {mem}")

from unsloth_candle import FastLanguageModel

# 1. Load model structure
print_memory("Initial")
model_name = "unsloth/Llama-3.2-1B-Instruct"

t0 = time.time()
print(f"Loading {model_name}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=512,
    load_in_4bit=True,
)
print(f"Loaded in {time.time() - t0:.1f}s")
print_memory("After Load")

# 2. Enable Inference with 4bit KV Cache
FastLanguageModel.for_inference(model, kv_quantization="4bit", use_rotor=True)
print_memory("After Inference Mode Setup")

# 3. Generate 100 Tokens
question = "Explain the theory of relativity in exactly 100 words."
inputs = tokenizer.encode(question)

print(f"\nQuestion: {question}")
print("Generating loop...")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt=True)
_ = model.generate(
    input_ids=torch.tensor([inputs]),
    streamer=text_streamer,
    max_new_tokens=100
)

print("")
print_memory("After generating 100 tokens with 4bit KV Cache")
print("Done!")
