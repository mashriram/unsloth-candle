import unsloth_candle
import sys

try:
    print("Testing unsloth_candle...")
    model = unsloth_candle.FastLanguageModel.from_pretrained("unsloth/Llama-3-8b", max_seq_length=2048, load_in_4bit=True)
    print("Model loaded successfully.")
    
    # Test forward (dummy square op)
    input_data = [1.0, 2.0, 3.0]
    output = model.forward(input_data)
    print(f"Input: {input_data}")
    print(f"Output: {output}")
    
    assert output == [1.0, 4.0, 9.0]
    print("Verification passed!")
except Exception as e:
    print(f"Verification failed: {e}")
    sys.exit(1)
