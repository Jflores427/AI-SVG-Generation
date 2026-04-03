import os
import torch
import transformers.utils.hub
transformers.utils.hub.TRANSFORMERS_CACHE = os.getenv("HF_HOME", "~/.cache/huggingface/hub")
os.environ["WANDB_DISABLED"] = "true"

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset
import io
import cv2
import cairosvg
import numpy as np
from PIL import Image
import editdistance
import xml.etree.ElementTree as ET
from skimage.metrics import structural_similarity as ssim

from transformers.trainer_utils import get_last_checkpoint

# ==========================================
# REWARD FUNCTIONS
# ==========================================
import re

def extract_svg(text):
    """Safely extracts the SVG from markdown backticks."""
    match = re.search(r'(<svg.*?</svg>)', str(text), re.IGNORECASE | re.DOTALL)
    return match.group(1) if match else str(text)

def render_to_numpy(svg_string):
    try:
        svg_string = extract_svg(svg_string) # Clean the string first!
        if not isinstance(svg_string, str) or "<svg" not in svg_string: return None
        # png_data = cairosvg.svg2png(bytestring=svg_string.encode('utf-8'), output_width=200, output_height=200, background_color="white")
        # OPTIMIZED:
        png_data = cairosvg.svg2png(bytestring=svg_string.encode('utf-8'), output_width=160, output_height=160, background_color="white")
        return np.array(Image.open(io.BytesIO(png_data)).convert('L'))
    except Exception: 
        return None

def calculate_ted(generated_svg, target_svg):
    def extract_tag_sequence(svg_string):
        try:
            root = ET.fromstring(svg_string)
            return [elem.tag.split('}')[-1] for elem in root.iter()]
        except Exception: 
            return []
    tags_gen = extract_tag_sequence(extract_svg(generated_svg))
    tags_tgt = extract_tag_sequence(target_svg)
    if not tags_gen or not tags_tgt: return 1000.0
    return float(editdistance.eval(tags_gen, tags_tgt))

def visual_similarity_reward(prompts, completions, ground_truths, **kwargs):
    rewards = []
    for completion, tgt_svg in zip(completions, ground_truths):
        # Safety catch: TRL sometimes passes strings, sometimes lists of dicts
        gen_text = completion[0]['content'] if isinstance(completion, list) else completion
        try:
            gen_img = render_to_numpy(gen_text)
            tgt_img = render_to_numpy(tgt_svg)
            if gen_img is None or tgt_img is None:
                rewards.append(0.0)
                continue
            ssim_score = ssim(tgt_img, gen_img, data_range=255)
            edges_tgt = cv2.Canny(tgt_img, 100, 200) > 0
            edges_gen = cv2.Canny(gen_img, 100, 200) > 0
            tp = np.logical_and(edges_tgt, edges_gen).sum()
            fp = np.logical_and(np.logical_not(edges_tgt), edges_gen).sum()
            fn = np.logical_and(edges_tgt, np.logical_not(edges_gen)).sum()
            f1 = (2 * tp / (2 * tp + fp + fn)) if (tp + fp + fn) > 0 else 0.0
            
            final_score = ((ssim_score + f1) / 2.0) * 0.85
            rewards.append(float(final_score))
        except Exception: 
            rewards.append(0.0)
    return rewards

def structural_reward(prompts, completions, ground_truths, **kwargs):
    rewards = []
    for completion, tgt_svg in zip(completions, ground_truths):
        gen_text = completion[0]['content'] if isinstance(completion, list) else completion
        try:
            ted = calculate_ted(gen_text, tgt_svg)
            s_score = np.exp(-ted / 25.0)
            rewards.append(float(s_score * 0.12))
        except Exception: 
            rewards.append(0.0)
    return rewards

def syntax_survival_reward(prompts, completions, **kwargs):
    rewards = []
    for completion in completions:
        gen_text = completion[0]['content'] if isinstance(completion, list) else completion
        try:
            # Extract the raw SVG from the markdown blocks
            clean_svg = extract_svg(gen_text).strip()
            if clean_svg.startswith("<svg") and clean_svg.endswith("</svg>"): 
                rewards.append(0.1)
            else: 
                rewards.append(-1.0)
        except Exception: 
            rewards.append(-1.0)
    return rewards

# ==========================================
# DATA FORMATTING
# ==========================================
def format_for_grpo(example):
    open_comment = "<!-" + "-"
    close_comment = "-" + "->"
    # Exact match to SFT prompt!
    prompt = f"{open_comment} Description: {example['prompt']} {close_comment}\n```xml\n"
    return {
        "prompt": [{"role": "user", "content": prompt}], 
        "ground_truths": example["svg"]
    }

# ==========================================
# MAIN TRAINING LOOP
# ==========================================
if __name__ == "__main__":
    MODEL_ID = "./sft-merged-model-3b" 
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": local_rank}, 
        torch_dtype=torch.float16,
    )
    model.warnings_issued = {}
    model.config.use_cache = True
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "left" 
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Satisfy the GRPOTrainer chat_template requirement 
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"

    # The New GRPO LoRA Adapter (Matches SFT size!)
    grpo_peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Load the highly optimized dataset
    train_optimized_filepath = "/kaggle/input/datasets/jef9921/train-optimized/train_optimized.parquet"
    raw_dataset = load_dataset("parquet", data_files=train_optimized_filepath, split="train")
    
    # We select 1000 samples. GRPO learns heavily from trial-and-error, so 1k is plenty!
    # grpo_dataset = raw_dataset.map(format_for_grpo, remove_columns=raw_dataset.column_names).select(range(1000))
    
    # We select 500 samples due to training time constraints. GRPO learns heavily from trial-and-error, so 500 is good enough! (Not in this case)
    grpo_dataset = raw_dataset.map(format_for_grpo, remove_columns=raw_dataset.column_names).select(range(500))

    rl_config = GRPOConfig(
        output_dir="./svg-phase2-rl",
        
        # Advanced Training Dynamics 
        learning_rate=1e-5,               # Keep this low! RL needs a small LR.
        lr_scheduler_type="cosine",       # Smooth decay
        warmup_steps=25,                  # 10% of max_steps for a gentle start
        weight_decay=0.01,                # Regularization
        optim="paged_adamw_8bit",         # CRITICAL: Brings back SFT VRAM savings

        # # Went Down from 32 -> 16, Not enough time
        # # Went Down From 16 - 8, ran into DDP-timeout error, GPU1 was probably waiting for GPU0 for longer than 30 minutes; Crashed
        # # 8 Generations divided by 2 GPUs = 4 generations per GPU (VRAM Safe!)
        # num_generations=8, 
        # per_device_train_batch_size=1, 
        # gradient_accumulation_steps=4,    
        
        # max_prompt_length=256, 
        # # max_completion_length=1024, 
        # max_completion_length=768, 
        # max_steps=250, 

        # SPEED TWEAKS 
        num_generations=4, 
        per_device_train_batch_size=1, 
        # gradient_accumulation_steps=4,  # Drops time-per-step by 50%
        gradient_accumulation_steps=2,  # Drops time-per-step by 50% again
        
        max_prompt_length=256, 
        max_completion_length=2048, 
        max_steps=125,                  # Caps total time to ~11 hours
        
        fp16=True,  
        bf16=False, 
        logging_steps=10,
        
        # CRITICAL DDP FIXES
        # The Multi-GPU Safety Net
        # Tells PyTorch to wait up to 1.5 hours (5400 seconds) for the other GPU
        # instead of killing the run after 30 minutes.
        ddp_timeout=5400,
        ddp_find_unused_parameters=False,
        use_vllm=False, 

        # THE CHECKPOINTING TWEAKS 
        save_strategy="steps",
        save_steps=25,
        save_total_limit=1,  # CRITICAL: Keeps only the most recent save to prevent Kaggle disk OOM!
    )

    grpo_trainer = GRPOTrainer(
        model=model,
        reward_funcs=[visual_similarity_reward, structural_reward, syntax_survival_reward], 
        args=rl_config,
        train_dataset=grpo_dataset,
        peft_config=grpo_peft_config, 
        # Pass our patched tokenizer to the trainer! 
        processing_class=tokenizer
    )

    # THE RESUME LOGIC 
    last_checkpoint = None
    if os.path.isdir(rl_config.output_dir):
        last_checkpoint = get_last_checkpoint(rl_config.output_dir)

    print(f"[GPU {local_rank}] Starting GRPO Training...")
    
    if last_checkpoint is not None:
        print(f"[GPU {local_rank}] Resuming from crashed checkpoint: {last_checkpoint}")
        grpo_trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        print(f"[GPU {local_rank}] Starting fresh training run.")
        grpo_trainer.train()
    
    if local_rank == 0:
        grpo_trainer.save_model("./grpo-lora-adapter-final")
        print("GRPO Phase 2 Complete!")
