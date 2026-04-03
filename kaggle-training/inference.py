import os
import argparse
import pandas as pd
import torch
import io
import cairosvg
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
import warnings
import re

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoModel
import outlines
from outlines.types import CFG

import time
from datetime import datetime
import sys

warnings.filterwarnings("ignore", category=UserWarning, message=".*Error in LLMatcher.*")

# ARGUMENT PARSER 
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use (0 or 1)")
parser.add_argument("--part", type=int, default=0, help="Part ID to use")
parser.add_argument("--start", type=int, default=0, help="Start index of the dataset")
parser.add_argument("--end", type=int, default=500, help="End index of the dataset")
args = parser.parse_args()

print(f"[GPU {args.gpu}] Initializing local inference.")

# Force the script to exclusively use the assigned GPU
device = f"cuda:{args.gpu}" 

# ==========================================
# LOAD MAIN MODEL (16-BIT PRECISION)
# ==========================================
print(f"[GPU {args.gpu}] Loading 3B model in 16-bit (Float16).")
hf_model = AutoModelForCausalLM.from_pretrained(
    "./sft-merged-model-3b-grpo", 
    torch_dtype=torch.float16, 
    device_map=device,
)

hf_model.generation_config.max_length = None
hf_tokenizer = AutoTokenizer.from_pretrained("./sft-merged-model-3b-grpo")
generator = outlines.from_transformers(hf_model, hf_tokenizer)

official_kaggle_grammar = CFG("""
    ?start: WS? svg WS?
    svg: "<svg" WS? ATTR_LIST ">" WS? elements "</svg>"
    elements: (element | TEXT)*
    element: "<" TAG WS? ATTR_LIST "/>" WS? | "<" TAG WS? ATTR_LIST ">" WS? elements "</" TAG ">" WS?
    TAG: "svg" | "g" | "path" | "rect" | "circle" | "ellipse" | "line" | "polyline" | "polygon" | "defs" | "use" | "symbol" | "clipPath" | "mask" | "linearGradient" | "radialGradient" | "stop" | "text" | "tspan" | "title" | "desc" | "style" | "pattern" | "marker" | "filter"
    ATTR_LIST: (/[a-zA-Z0-9_:-]+/ WS? "=" WS? /"[^"]*"/ WS?)*
    TEXT: /[^<]+/
    WS: /[ \\t\\n\\r]+/
""")

# ==========================================
# LOAD SIGLIP JUDGE (16-BIT PRECISION)
# ==========================================
print(f"[GPU {args.gpu}] Loading SigLIP Judge.")
siglip_id = "google/siglip-so400m-patch14-384"
processor = AutoProcessor.from_pretrained(siglip_id)

judge = AutoModel.from_pretrained(
    siglip_id, 
    torch_dtype=torch.float16 
).to(device)
judge.eval() 

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def extract_svg(text):
    match = re.search(r'(<svg.*?</svg>)', text, re.IGNORECASE | re.DOTALL)
    return match.group(1) if match else text

def heal_svg(raw_svg):
    if raw_svg.strip().endswith("</svg>"): return raw_svg
    last_closed_idx = raw_svg.rfind("/>")
    if last_closed_idx != -1:
        return raw_svg[:last_closed_idx + 2] + "\n</svg>"
    return raw_svg

def is_kaggle_compliant(svg_string):
    if len(svg_string) > 16000: return False
    try:
        root = ET.fromstring(svg_string)
        if root.tag.split('}')[-1] != 'svg': return False
    except ET.ParseError: return False
    if svg_string.count("<path") > 256: return False
    return True

def render_to_numpy(svg_string):
    try:
        png_data = cairosvg.svg2png(bytestring=svg_string.encode('utf-8'), output_width=256, output_height=256, background_color="white")
        return np.array(Image.open(io.BytesIO(png_data)).convert('L'))
    except: return None

def select_best_svg(prompt_text, candidate_svgs):
    valid_images, valid_svgs = [], []
    for i, raw_svg in enumerate(candidate_svgs):
        clean_svg = heal_svg(extract_svg(raw_svg))
        if not is_kaggle_compliant(clean_svg): continue
        img = render_to_numpy(clean_svg)
        if img is None: continue
        valid_images.append(Image.fromarray(img).convert('RGB'))
        valid_svgs.append(clean_svg) 
            
    if not valid_images: return "<svg></svg>"
    if len(valid_images) == 1: return valid_svgs[0]

    inputs = processor(
        text=[prompt_text], images=valid_images, return_tensors="pt", 
        padding="max_length", truncation=True, max_length=64    
    ).to(device)
    
    with torch.no_grad():
        scores = judge(**inputs).logits_per_image.squeeze().cpu().numpy()
        
    if scores.ndim == 0: return valid_svgs[0]
    return valid_svgs[scores.argmax()]

# ==========================================
# INFERENCE LOOP
# ==========================================
if __name__ == "__main__":
    # SLICE THE DATASET 
    # full_df = pd.read_csv("dl-spring-2026-svg-generation/test.csv") 
    full_df = pd.read_csv("/kaggle/input/competitions/dl-spring-2026-svg-generation/test.csv") 
    test_df = full_df.iloc[args.start:args.end].copy()
    
    csv_filename = f"./submission-3b-2048-grpo_part_{args.part}.csv"
    NUM_CANDIDATES = 2
    MAX_CONTEXT_WINDOW = 2048
    
    print(f"\n[GPU {args.gpu}] Starting Inference on {len(test_df)} prompts (Rows {args.start} to {args.end})...")
    sys.stdout.flush() # Force write to file!

    for idx in range(len(test_df)):
        # Start the stopwatch!
        start_time = time.time()
        current_time = datetime.now().strftime("%H:%M:%S")
        
        row = test_df.iloc[idx]
        print(f"\n[{current_time}] [GPU {args.gpu}] [{idx+1}/{len(test_df)}] Generating ID: {row['id']} | Prompt: {row['prompt'][:40]}...")
        sys.stdout.flush() 
        
        open_comment = "<!-" + "-"
        close_comment = "-" + "->"
        prompt_text = f"{open_comment} Description: {row['prompt']} {close_comment}\n```xml\n"
        
        inputs = hf_tokenizer(prompt_text, return_tensors="pt")
        prompt_token_count = inputs["input_ids"].shape[1]
        safe_max_new_tokens = MAX_CONTEXT_WINDOW - prompt_token_count - 10

        candidates = [] 
        
        try:
            raw_candidate = generator(
                prompt_text, 
                official_kaggle_grammar, 
                do_sample=True, 
                temperature=0.7, 
                max_new_tokens=safe_max_new_tokens,
                num_return_sequences=NUM_CANDIDATES, 
            )
            
            if isinstance(raw_candidate, list):
                candidates = raw_candidate
            else:
                candidates.append(raw_candidate)
                
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [GPU {args.gpu}] Generation failed: {e}")
            sys.stdout.flush()
            
        torch.cuda.empty_cache() 
            
        best_svg = select_best_svg(row["prompt"], candidates)

        print(f"Best SVG: {best_svg}")
        sys.stdout.flush()
        
        result_df = pd.DataFrame([{"id": row["id"], "svg": best_svg}])
        if idx == 0 or not os.path.exists(csv_filename):
            result_df.to_csv(csv_filename, index=False, mode='w')
        else:
            result_df.to_csv(csv_filename, index=False, mode='a', header=False)
            
        torch.cuda.empty_cache()

        # Stop the stopwatch and log the duration!
        elapsed_time = time.time() - start_time
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [GPU {args.gpu}] Finished prompt {idx+1} in {elapsed_time:.2f} seconds.")
        sys.stdout.flush()

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] [GPU {args.gpu}] Local inference complete! Saved to {csv_filename}")
    sys.stdout.flush()
