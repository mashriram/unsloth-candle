import unsloth_candle
import sys

# Using TinyLlama
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print(f"Testing Trainer with {MODEL_NAME}...")

try:
    # 1. Load Model
    model = unsloth_candle.FastLanguageModel.from_pretrained(
        MODEL_NAME, 
        max_seq_length=2048, 
        load_in_4bit=True
    )
    print("Model loaded successfully!")
    
    # 2. Create Trainer
    trainer = unsloth_candle.Trainer(model)
    trainer.configure_optimizer(1e-3) # High LR to see change quickly
    print("Trainer initialized.")
    
    # 3. Training Loop
    losses = []
    print("Starting training loop...")
    for i in range(10):
        # inputs are dummy IDs
        loss = trainer.train_step([1, 2, 3, 4, 5])
        print(f"Step {i}: Loss = {loss}")
        losses.append(loss)
        
    # Check if loss decreases
    if losses[-1] < losses[0]:
        print("Success: Loss decreased!")
    else:
        print("Failure: Loss did not decrease.")
        # It might not decrease monotonically due to noise or structure, but with 1e-3 and MSE(logits, 0) it should.
        # Initial logits are likely non-zero. Bias is initialized to random.
        # If bias drives logits to 0, loss should drop.
        sys.exit(1)

except Exception as e:
    print(f"Training failed: {e}")
    sys.exit(1)
