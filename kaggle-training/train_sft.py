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
    
    # Split into Train and Eval 
    dataset_split = raw_dataset.select(range(25000)).train_test_split(test_size=500, seed=25)
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]
    
    MODEL_ID = "unsloth/Qwen2.5-Coder-3B"
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=2048,
        dtype=torch.float16, 
        load_in_4bit=True,
    )
    
    model = FastLanguageModel.get_peft_model(
        model, 
        # Increased Capacity & MLP Layers 
        r=32, 
        lora_alpha=64, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
        lora_dropout=0, 
        bias="none",
        use_gradient_checkpointing=True,
        random_state=25,
    )
    
    EOS = tokenizer.eos_token
    
    def format_for_sft(example):
        open_comment = "<!-" + "-"
        close_comment = "-" + "->"
        text = f"{open_comment} Description: {example['prompt']} {close_comment}\n```xml\n{example['svg']}\n```{EOS}"
        return {"text": text}
    
    train_dataset = train_dataset.map(format_for_sft)
    eval_dataset = eval_dataset.map(format_for_sft)
    
    sft_config = SFTConfig(
        output_dir="./svg-phase1-sft",
        dataset_text_field="text",
        max_seq_length=2048,
        
        # Dataset Packing 
        packing=True, 
        
        num_train_epochs=1, 
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8, 
        
        # Advanced Training Dynamics 
        learning_rate=2e-4,
        lr_scheduler_type="cosine", 
        warmup_steps=100,           
        weight_decay=0.01,          
        neftune_noise_alpha=5,      
        
        fp16=True,  
        bf16=False, 
        optim="paged_adamw_8bit", 
        logging_steps=50,
        
        # Evaluation Strategy 
        eval_strategy="steps",
        eval_steps=200, 
        
        # Save exactly when you evaluate, and keep the best one! 
        save_strategy="steps", 
        save_steps=200,
        load_best_model_at_end=True,         # Auto-reverts to the best checkpoint!
        metric_for_best_model="eval_loss",   # Uses your eval set as the judge
        greater_is_better=False,             # Because lower loss is better
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
    
    print("Starting SFT Phase 1 with Qwen2.5-Coder-3B...")
    sft_trainer.train()
    
    model.save_pretrained("./sft-lora-adapter-3B")
    tokenizer.save_pretrained("./sft-lora-adapter-3B")
