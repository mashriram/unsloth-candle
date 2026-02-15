import unsloth_candle
import sys

# Using TinyLlama to avoid huge download/gated access issues during CI/Dev
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# MODEL_NAME = "unsloth/Llama-3-8b" # Uncomment to test real target if available

print(f"Testing unsloth_candle with {MODEL_NAME}...")

try:
    # 1. Load Model
    model = unsloth_candle.FastLanguageModel.from_pretrained(
        MODEL_NAME, 
        max_seq_length=2048, 
        load_in_4bit=True
    )
    print("Model loaded successfully!")
    
    # 2. Forward Pass
    # Dummy input: [1, 2, 3]
    print("Running forward pass...")
    output = model.forward([1, 2, 3])
    print(f"Forward pass output: {output}")
    
    assert output == "Forward pass success"
    print("Verification passed!")
except Exception as e:
    print(f"Verification failed: {e}")
    sys.exit(1)
