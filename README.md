Here is a comprehensive `README.md` file based on the setup, execution flow, and prerequisites detailed in your Jupyter Notebook. 

***

```markdown
# Deep Learning SFT 3B (2-Epoch) - SVG Generation

This repository contains the end-to-end Jupyter Notebook (`Deep_Learning_SFT_3B_2048_Epoch2_Final.ipynb`) for training and inferencing a 3-Billion parameter language model (Qwen2.5-Coder-3B) to generate SVG graphics from text prompts. 

The pipeline includes a "Data Diet" optimization step, a two-epoch Supervised Fine-Tuning (SFT) strategy using Unsloth, and a robust multi-pass "rescue" inference cascade utilizing a SigLIP judge to ensure high-quality, competition-compliant SVG outputs.

## 📋 Prerequisites

Before running the notebook, ensure you have the following ready:

1. **Hardware:** A GPU with at least 8GB-15GB VRAM (e.g., NVIDIA T4, RTX 3090, or Kaggle/Colab cloud environments). *Note: The model merging step is specifically routed to the CPU to prevent VRAM Out-Of-Memory (OOM) crashes on smaller GPUs.*
2. **Kaggle API Token:** You must have your `kaggle.json` file placed in the root directory to download the competition dataset (`dl-spring-2026-svg-generation-from-text-prompts-extended-deadline`).
3. **Python Environment:** Python 3.10+ is recommended. 

## 🛠️ Installation & Setup

It is highly recommended to run this inside a virtual environment.

**1. Create and activate a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

**2. Install the required dependencies:**
Run the following pip command to install all necessary packages (including Unsloth, PyTorch, and image processing tools):
```bash
pip install -q pandas numpy torch pathlib kaggle
pip install git+https://github.com/unslothai/unsloth-zoo.git git+https://github.com/unslothai/unsloth.git trl==0.24.0 bitsandbytes picosvg cairosvg opencv-python-headless editdistance mergekit llm_blender outlines scikit-learn weave wandb protobuf llguidance
```

**3. Setup Kaggle Credentials:**
Place your `kaggle.json` file in the same folder as the notebook, or set it up globally:
```bash
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

## 🚀 Usage Guide

The notebook is divided into three main phases: **Data Analysis/Optimization**, **Training**, and **Inference**. 

### Phase 1: Data Preparation
Run the early cells of the notebook to automatically download the Kaggle dataset and unpack it into `./dl-spring-2026-svg-generation`. 
Next, run the **Data Diet** section. This will heavily compress, format, and optimize the raw SVGs using `picosvg` and regex regex functions, saving the result as `train_optimized.parquet` (this takes ~20 minutes on 4 CPU cores).

### Phase 2: Training (Optional if you already have the weights)
*Note: The training code blocks are commented out by default in the notebook. If you wish to train the model from scratch, uncomment the respective cells.*

1. **Epoch 1:** Trains an initial LoRA adapter (`./sft-lora-adapter-3B`) on the first 25,000 samples and merges it into the base model (`./sft-merged-model-3b`).
2. **Epoch 2 (Warm Restart):** Loads the merged model and trains a *new* LoRA adapter (`./sft-lora-adapter-3B-epoch2`) on the remaining ~24,000 unseen samples with a lowered learning rate.
3. **Final Merge:** Bakes the Epoch 2 adapter into the base weights to create the `./final-merged-epoch2` model.

### Phase 3: Inference (Generation & Rescue Loop)
If you already have `./final-merged-epoch2` generated, you can skip directly to the **Inference Section**. 

The inference engine runs a dynamic **Rescue Cascade** using `outlines` Context-Free Grammar (CFG) and a `google/siglip-so400m-patch14-384` visual judge:
1. **Passes:** Attempts generation with varying count of token context window.
2. **Healer Check:** Any truncated or malformed SVGs are repaired using a custom Quote-Aware LIFO stack algorithm (`heal_svg_updated`) and verified for valid `<path>` geometric structures.
3. **Rescue Pass:** Prompts that yield blank canvases (`<svg></svg>`) are automatically passed down to higher-budget token/candidate rescue passes.
4. **Outputs:** The notebook automatically saves results incrementally to `.csv` files (e.g., `submission_hallucination-3b-2048-p2.csv`).

## 📁 Directory Structure Overview

By the end of a full end-to-end run, your working directory could look like this:

```text
├── Deep_Learning_SFT_3B_2048_Epoch2_Final.ipynb
├── kaggle.json
├── train_optimized.parquet               # Generated via Data Diet
├── dl-spring-2026-svg-generation/        # Unzipped Kaggle dataset
│   ├── train.csv
│   └── test.csv
├── sft-lora-adapter-3B/                  # Output of Epoch 1
├── sft-merged-model-3b/                  # Base model + Epoch 1
├── sft-lora-adapter-3B-epoch2/           # Output of Epoch 2
├── final-merged-epoch2/                  # FINAL INFERENCE MODEL
└── submission_*.csv                      # Incrementally saved Kaggle submissions
```

## ⚠️ Known Warnings
* **Tokenizer Indexing Warnings:** You may see a warning during the token-length calculation phase (`Token indices sequence length is longer than the specified maximum sequence length...`). This is normal and purely analytical; the actual inference loop dynamically prevents out-of-bounds context window errors.
* **CPU Merging:** The model merging cells explicitly use `device_map="cpu"` to prevent VRAM fragmentation. Ensure your system has sufficient regular RAM (~16GB+) during this step.