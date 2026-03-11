from dataclasses import dataclass, field
from typing import Optional, List, Any
from .unsloth_candle import Trainer as RustTrainer

@dataclass
class SFTConfig:
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 5
    max_steps: int = 60
    learning_rate: float = 2e-4
    logging_steps: int = 1
    optim: str = "adamw_8bit"
    weight_decay: float = 0.001
    lr_scheduler_type: str = "linear"
    seed: int = 3407
    output_dir: str = "outputs"
    report_to: str = "none"

class SFTTrainer:
    def __init__(
        self,
        model,
        tokenizer=None,
        train_dataset=None,
        eval_dataset=None,
        dataset_text_field: str = "text",
        max_seq_length: int = 2048,
        data_collator=None,
        packing: bool = False,
        args: Optional[SFTConfig] = None,
        **kwargs
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.args = args or SFTConfig()
        
        # Initialize the Rust trainer
        # Note: model.rust_flm is the actual Rust object
        self.rust_trainer = RustTrainer(model.rust_flm)
        self.rust_trainer.configure_optimizer(self.args.learning_rate)

    def train(self):
        print(f"Starting training for {self.args.max_steps} steps...")
        
        import time
        
        # Get samples from dataset if available
        samples = []
        if self.train_dataset is not None:
            # Basic extraction for demo purposes
            for i in range(min(len(self.train_dataset), self.args.max_steps)):
                # Handle both list and dataset-like objects
                try:
                    item = self.train_dataset[i]
                except:
                    continue
                    
                text = item.get("text", "")
                if text and self.tokenizer:
                    tokens = self.tokenizer.encode(text)
                    samples.append(tokens)

        # Fallback to dummy data if dataset is empty or not provided
        if not samples:
            samples = [[1, 2, 3, 4]]

        all_losses = []
        for step in range(self.args.max_steps):
            t0 = time.perf_counter()
            sample = samples[step % len(samples)]
            loss = self.rust_trainer.train_step(sample)
            all_losses.append(loss)
            elapsed = (time.perf_counter() - t0) * 1000
            
            if step % self.args.logging_steps == 0:
                print(f"Step {step+1}/{self.args.max_steps} - loss: {loss:.4f} - {elapsed:.0f}ms")
        
        print("Training complete.")
        return {
            "train_runtime": 0.0, 
            "train_samples_per_second": 0.0, 
            "total_flos": 0.0, 
            "train_loss": loss,
            "losses": all_losses
        }
