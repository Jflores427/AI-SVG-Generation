import os
import torch
from unsloth import FastLanguageModel

import transformers.utils.hub
transformers.utils.hub.TRANSFORMERS_CACHE = os.getenv("HF_HOME", "~/.cache/huggingface/hub")
os.environ["WANDB_DISABLED"] = "true"

from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

if __name__ == "__main__":
    train_optimized_filepath = "/kaggle/input/datasets/jef9921/train-optimized/train_optimized.parquet"
    raw_dataset = load_dataset("parquet", data_files=train_optimized_filepath, split="train")
    
    # THE DATA ALPHA STRATEGY 
    # Instead of repeating the first 25,000, we feed it the REMAINING 24,043 SVGs!
    # The model learns entirely new concepts it has never seen before.
    dataset_split = raw_dataset.select(range(25000, 49043)).train_test_split(test_size=500, seed=42)
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]
    
    # Load the 16-bit merged model 
    MODEL_ID = "./sft-merged-model-3b" 
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=2048,
        dtype=torch.float16, 
        load_in_4bit=True,
    )
    
    # We apply a fresh LoRA adapter to capture these new, refined details
    model = FastLanguageModel.get_peft_model(
        model, 
        r=32, 
        lora_alpha=64, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
        lora_dropout=0, 
        bias="none",
        use_gradient_checkpointing=True,
        random_state=42, # Changed seed to shake up the data ordering
    )
    
    EOS = tokenizer.eos_token
    
    # Format the dataset for SFT by combining prompt and optimized SVG into a single text field
    def format_for_sft(example):
        open_comment = "<!-" + "-"
        close_comment = "-" + "->"
        text = f"{open_comment} Description: {example['prompt']} {close_comment}\n```xml\n{example['svg']}\n```{EOS}"
        return {"text": text}
    
    train_dataset = train_dataset.map(format_for_sft)
    eval_dataset = eval_dataset.map(format_for_sft)
    
    sft_config = SFTConfig(
        output_dir="./svg-phase2-sft-epoch2", # Save to a new folder!
        dataset_text_field="text",
        max_seq_length=2048,
        
        # Packing Enabled to Squeeze More Learning Signal
        packing=True, 
        
        num_train_epochs=1, 
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8, 
        
        # THE WARM RESTART DYNAMICS
        learning_rate=5e-5,          # Dropped from 2e-4 to gently refine the weights
        lr_scheduler_type="cosine", 
        warmup_ratio=0.05,           # 5% warmup for the new curve
        weight_decay=0.01,          
        neftune_noise_alpha=5,      
        
        fp16=True,  
        bf16=False, 
        optim="paged_adamw_8bit", 
        logging_steps=50,
        
        eval_strategy="steps",
        eval_steps=200, 
        
        save_strategy="steps", 
        save_steps=200,
        load_best_model_at_end=True,         
        metric_for_best_model="eval_loss",   
        greater_is_better=False,             
        save_total_limit=2,
        ddp_find_unused_parameters=False,
    )
    
    sft_trainer = SFTTrainer(
        model=model, 
        tokenizer=tokenizer, 
        train_dataset=train_dataset, 
        eval_dataset=eval_dataset,
        args=sft_config
    )
    
    print("Starting SFT Epoch 2 (Part 2) with Merged Qwen2.5-Coder-3B...")
    sft_trainer.train()
    
    # Save the new adapter!
    model.save_pretrained("./sft-lora-adapter-3B-epoch2")
    tokenizer.save_pretrained("./sft-lora-adapter-3B-epoch2")
