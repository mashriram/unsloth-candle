import os
import torch
import torch.nn as nn
from .unsloth_candle import FastLanguageModel as RustFLM
import unsloth_candle.unsloth_candle as rust_module
RustFLM = rust_module.FastLanguageModel

class ModelWrapper(nn.Module):
    def __init__(self, rust_flm):
        super().__init__()
        self.rust_flm = rust_flm
        # Unsloth models often have these attributes
        self.max_seq_length = 2048 
        self.is_inference = False

    def forward(self, input_ids, **kwargs):
        # Compatibility with torch/hf style forward
        if torch.is_tensor(input_ids):
            ids = input_ids.flatten().tolist()
        else:
            ids = list(input_ids)
        
        # Rust forward returns next token ID for now
        return self.rust_flm.forward(ids, pos=0)

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens=128,
        streamer=None,
        use_cache=True,
        temperature=1.0,
        min_p=0.0,
        **kwargs
    ):
        if torch.is_tensor(input_ids):
            current_ids = input_ids.flatten().tolist()
        else:
            current_ids = list(input_ids)

        # First pass: process the whole prompt
        next_token = self.rust_flm.forward(current_ids, pos=0)
        generated = [next_token]
        
        if streamer:
            streamer.put(torch.tensor([[next_token]]))

        # Generation loop with KV caching
        eos_token_id = kwargs.get("eos_token_id", 128001)

        for i in range(1, max_new_tokens):
            if next_token == eos_token_id:
                break
                
            # Efficiently pass only the last token and the current position
            pos = len(current_ids) + len(generated) - 1
            next_token = self.rust_flm.forward([next_token], pos=pos)
            generated.append(next_token)
            
            if streamer:
                streamer.put(torch.tensor([[next_token]]))
                
        if streamer:
            streamer.end()
            
        return torch.tensor([current_ids + generated])

    def save_pretrained_merged(self, output_dir, tokenizer=None, save_method="merged_16bit", **kwargs):
        print(f"Saving merged model to {output_dir} (method: {save_method})...")
        path = self.rust_flm.save_pretrained_merged(output_dir)
        if tokenizer:
            tokenizer.save_pretrained(output_dir)
        return path

    def save_pretrained_gguf(self, output_dir, tokenizer=None, quantization_type="q8_0", **kwargs):
        print(f"Saving GGUF model to {output_dir} ({quantization_type})...")
        path = self.rust_flm.save_to_gguf(output_dir, quantization_type)
        if tokenizer:
            tokenizer.save_pretrained(output_dir)
        return path

    def save_in_4bit(self, output_dir, tokenizer=None, **kwargs):
        print(f"Saving 4-bit NF4 model to {output_dir}...")
        path = self.rust_flm.save_in_4bit(output_dir)
        if tokenizer:
            tokenizer.save_pretrained(output_dir)
        return path

    def push_to_hub_merged(self, repo_id, tokenizer, save_method="merged_16bit", token=None, **kwargs):
        temp_dir = "./temp_hub_save"
        self.save_pretrained_merged(temp_dir, tokenizer, save_method)
        print(f"Pushing to hub: {repo_id}...")
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.upload_folder(folder_path=temp_dir, repo_id=repo_id, repo_type="model")

class FastLanguageModel:
    @staticmethod
    def from_pretrained(
        model_name,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
        token=None,
        **kwargs
    ):
        # Note: Rust implementation returns (flm, tokenizer)
        rust_flm, tokenizer = RustFLM.from_pretrained(
            model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            token=token
        )
        model = ModelWrapper(rust_flm)
        model.max_seq_length = max_seq_length
        return model, tokenizer

    @staticmethod
    def get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=16,
        lora_dropout=0,
        use_gradient_checkpointing=True,
        **kwargs
    ):
        model.rust_flm.apply_lora(
            target_modules=target_modules,
            rank=r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            use_dora=kwargs.get("use_dora", False)
        )
        return model

    @staticmethod
    def for_inference(model):
        model.rust_flm.for_inference()
        model.is_inference = True
