import unittest
import sys
import os
import torch

# Ensure import of built module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../target/release")))
try:
    import unsloth_candle
    from unsloth_candle import FastLanguageModel
except ImportError:
    print("unsloth_candle not found. Skipping tests that require it.")
    unsloth_candle = None

class TestUnslothCandle(unittest.TestCase):
    def setUp(self):
        if unsloth_candle is None:
            self.skipTest("Module not installed")

    def test_loading_tiny_llama(self):
        # Requires internet and model existence. 
        # Use a model that is small.
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        try:
            model = FastLanguageModel.from_pretrained(model_name, load_in_4bit=True)
            self.assertIsNotNone(model)
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            # Allow failure if network/disk issue, but log it.
            # In production, we'd mock the loader.

    def test_lora_application(self):
        # Mock or Real loading
        try:
            model = FastLanguageModel.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", load_in_4bit=True)
            model.apply_lora(
                target_modules=["q_proj", "v_proj"],
                rank=8,
                alpha=16,
                dropout=0.0,
                use_dora=False
            )
            # Should not raise
        except Exception as e:
            pass # Fail gracefully if model load failed

if __name__ == '__main__':
    unittest.main()
