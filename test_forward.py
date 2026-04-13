import torch
from unsloth_candle import FastLanguageModel

model_name = "unsloth/Llama-3.2-1B-Instruct"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=128,
    load_in_4bit=True,
)

print("Running forward pass...")
tokens = tokenizer.encode("Hello, how are you?")
next_token = model.forward(tokens, pos=0)
print(f"Next token ID: {next_token}")
print("Forward pass successful!")
