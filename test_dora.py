import unsloth_candle
import time

def test_dora():
    model_name = "HuggingFaceTB/SmolLM-135M"
    print(f"Loading {model_name}...")
    model = unsloth_candle.FastLanguageModel.from_pretrained(model_name, None, None)
    print("Model loaded.")
    
    trainer = unsloth_candle.Trainer(model)
    trainer.configure_optimizer(learning_rate=1e-3)

    input_ids = [101, 2045, 2003, 2070, 102] # Random tokens
    
    print("Running initial forward pass...")
    res = model.forward(input_ids)
    print(f"Forward result: {res}")
    
    print("Applying DoRA to [q_proj, v_proj]...")
    targets = ["q_proj", "v_proj"]
    # use_dora=True
    model.apply_lora(targets, rank=4, alpha=8.0, dropout=0.0, use_dora=True)
    print("DoRA applied successfully.")
    
    print("Starting training loop (5 steps)...")
    for i in range(5):
        loss = trainer.train_step(input_ids)
        print(f"Step {i+1}: Loss = {loss}")

if __name__ == "__main__":
    test_dora()
