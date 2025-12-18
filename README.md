# MAD-X: Zero-Shot Cross-Lingual Transfer with Adapters

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Adapters](https://img.shields.io/badge/Adapters-Hub-green)
![Status](https://img.shields.io/badge/Status-Functional-brightgreen)

## 📖 Project Overview

This repository contains a complete, reproducible implementation of the **MAD-X framework** (Multiple Adapters for Cross-lingual Transfer). The goal of this project is to enable **Zero-Shot Cross-Lingual Named Entity Recognition (NER)**.

Instead of fine-tuning the entire model (which is computationally expensive and prone to catastrophic forgetting), this project uses **Adapter Modules**—small, trainable layers inserted into a frozen pre-trained model (`XLM-Roberta`).

### The Pipeline
1.  **Source Language Adaptation:** Teach the model "English" grammar using an Invertible Adapter.
2.  **Task Adaptation:** Teach the model "NER" logic using a Task Adapter stacked on top.
3.  **Target Language Adaptation:** Teach the model a "Target" language (simulated with WikiText) using a separate Invertible Adapter.
4.  **Zero-Shot Transfer:** Swap the language adapters to perform NER on the target language without ever seeing labeled NER data for it.

---

## 🛠️ Architecture & Technical Stack

* **Base Model:** `xlm-roberta-base` (Frozen)
* **Adapter Config:**
    * *Language:* `pfeiffer+inv` (Invertible Adapters for vocabulary alignment)
    * *Task:* `pfeiffer` (Standard Bottleneck Adapters)
* **Datasets:**
    * Language Modeling: `wikitext-2-raw-v1`
    * NER Task: `conll2003`

---

## ⚙️ Installation & Dependencies

This project requires a specific environment configuration to resolve conflicts between `datasets`, `fsspec`, and `numpy` C-APIs.

```bash
# 1. Clone the repository
git clone [https://github.com/Ivar1331/MAD-X-Implementation.git](https://github.com/Ivar1331/MAD-X-Implementation.git)
cd MAD-X-Implementation

# 2. Install PyTorch (Ensure CUDA support if using GPU)
pip install torch torchvision torchaudio

# 3. Install the Adapters library (Editable mode)
pip install -e .

# 4. Install Critical Fixed-Version Dependencies
# These specific versions prevent the "Script not supported" and "PyExtensionType" errors
pip install datasets==2.14.0
pip install fsspec==2023.9.2
pip install pyarrow==14.0.1
pip install "numpy<2.0"

## 🚀 Usage Guide

**Note:** For servers with multiple GPUs, always use `CUDA_VISIBLE_DEVICES=0` to prevent tensor device mismatches.

### Stage 1: Train Source Language Adapter (English)
Trains an adapter on raw text to capture language structure.
```bash
CUDA_VISIBLE_DEVICES=0 python examples/pytorch/language-modeling/run_mlm.py \
  --model_name_or_path xlm-roberta-base \
  --train_adapter \
  --adapter_config "pfeiffer+inv" \
  --output_dir ./tmp/madx_language_adapter \
  --dataset_name wikitext \
  --dataset_config_name wikitext-2-raw-v1 \
  --do_train \
  --learning_rate 1e-4 \
  --num_train_epochs 3.0

### Stage 2: TTrain Task Adapter (NER)
CUDA_VISIBLE_DEVICES=0 python examples/pytorch/token-classification/run_ner.py \
  --model_name_or_path xlm-roberta-base \
  --train_adapter \
  --adapter_config "pfeiffer" \
  --load_adapter ./tmp/madx_language_adapter \
  --output_dir ./tmp/madx_task_adapter \
  --dataset_name conll2003 \
  --do_train \
  --learning_rate 1e-4 \
  --num_train_epochs 3.0

### Stage 3: Train Target Language Adapter
CUDA_VISIBLE_DEVICES=0 python examples/pytorch/language-modeling/run_mlm.py \
  --model_name_or_path xlm-roberta-base \
  --train_adapter \
  --adapter_config "pfeiffer+inv" \
  --output_dir ./tmp/madx_target_lang_adapter \
  --dataset_name wikitext \
  --dataset_config_name wikitext-2-raw-v1 \
  --do_train \
  --learning_rate 1e-4 \
  --num_train_epochs 3.0

### Stage 4: Zero-Shot Evaluation (The Final Result)
CUDA_VISIBLE_DEVICES=0 python examples/pytorch/token-classification/run_ner.py \
  --model_name_or_path xlm-roberta-base \
  --train_adapter \
  --do_eval \
  --dataset_name conll2003 \
  --load_adapter ./tmp/madx_target_lang_adapter \
  --load_adapter ./tmp/madx_task_adapter \
  --output_dir ./tmp/madx_zero_shot_eval

### Technical Challenges Solved
Dependency Hell Resolution:

Downgraded datasets to 2.14.0 to allow execution of local dataset loading scripts (deprecated in v3.0).

Pinned numpy<2.0 to fix binary incompatibility with pyarrow (the _ARRAY_API crash).

Codebase Patching:

Manually patched run_ner.py and run_mlm.py to remove the trust_remote_code argument, which caused crashes with older stable library versions.

Infrastructure Optimization:

Enforced Single-GPU training on Multi-GPU nodes to solve RuntimeError: mat2 device mismatch.
