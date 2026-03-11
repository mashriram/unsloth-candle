#!/usr/bin/env python3
import os
import sys
import time

# Ensure we can import the built library
sys.path.append(os.path.abspath("python"))

import unsloth_candle
from unsloth_candle import FastLanguageModel

def main():
    print("═" * 60)
    print("  unsloth-candle  Post-Fine-tuning Verification")
    print("═" * 60)

    # 1. Path to the saved merged model
    MODEL_DIR = "./output/hf_merged"
    if not os.path.exists(MODEL_DIR):
        print(f"ERROR: Merged model directory not found at {MODEL_DIR}")
        sys.exit(1)

    print(f"── STEP 1: Loading fine-tuned model from {MODEL_DIR} ──")
    
    t0 = time.perf_counter()
    # Loading the merged model as a regular model
    model, tokenizer = FastLanguageModel.from_pretrained(
        MODEL_DIR,
        load_in_4bit=False,
    )
    print(f"  ✓ Model loaded in {time.perf_counter() - t0:.1f}s")

    # 2. Test Question
    print(f"\n── STEP 2: Running test question ──")
    question = "Explain how neural networks learn in one sentence."
    print(f"  Question: {question}")

    # Tokenize input
    input_ids = tokenizer.encode(question)
    
    # Run forward pass (just to see if it works)
    print("  Running forward pass...")
    result = model.forward(input_ids)
    print(f"  ✓ Forward pass result: {result}")

    print(f"\n═{'═' * 60}")
    print("  ✅ Fine-tuned model verification complete!")
    print(f"═{'═' * 60}\n")

if __name__ == "__main__":
    main()
