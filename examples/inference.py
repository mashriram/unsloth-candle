import sys
import os
try:
    import unsloth_candle
except ImportError:
    print("unsloth_candle not found. Please run 'maturin develop' or install the package.")
    sys.exit(1)

from unsloth_candle import FastLanguageModel

def main():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    print(f"Loading {model_name}...")
    model = FastLanguageModel.from_pretrained(
        model_name,
        load_in_4bit=True
    )
    
    input_ids = [1, 306, 25] # "Hello world" ids placeholder
    
    # Simple forward pass
    print("Running inference (forward pass)...")
    res = model.forward(input_ids)
    print(f"Result: {res}")

if __name__ == "__main__":
    main()
