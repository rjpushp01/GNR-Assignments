# GNR 638 Assignment 2 — CNN Transfer Learning

This repository contains the implementation for GNR 638 Assignment 2, which systematically analyzes representation transfer, fine-tuning strategies, data efficiency under few-shot settings, and robustness to input corruption using pre-trained CNN backbones on the **Aerial Images Dataset (AID)**.

**Selected Models:**
- ResNet50
- Inception v3
- DenseNet121

---

## 🛠️ Setup Instructions

### 1. Environment Setup
It is recommended to use an isolated Python environment (e.g., conda or venv). The code is tested on Python 3.10+ and PyTorch 2.6.0+cu124.

```bash
conda create -n gnr638_env python=3.10
conda activate gnr638_env
```

### 2. Install Dependencies
Install all required libraries including PyTorch, `timm` (for pre-trained models), `ptflops` (for efficiency metrics), and `umap-learn`:

```bash
pip install -r requirements.txt
```

### 3. Dataset Preparation
Ensure the Aerial Images Dataset (AID) is structured in an ImageFolder format (i.e., each class has its own subdirectory). By default, the scripts expect the dataset to be in a directory named `train_data/` at the root of the project. 

If your dataset or hidden test set is located elsewhere, you can pass the path via the `--data_root` argument when running the scripts.

---

## 🚀 How to Run Models (Training & Evaluation)

A unified entry point script, `train.py`, is provided to execute all 5 experimental scenarios.

**General Usage:**
```bash
python train.py --model {resnet50, inception_v3, densenet121} --scenario {1,2,3,4,5} [--data_root /path/to/data]
```

### Scenario 1: Linear Probe Transfer
Freezes the backbone and trains only the linear classification head.
```bash
python train.py --model resnet50 --scenario 1
python train.py --model inception_v3 --scenario 1
python train.py --model densenet121 --scenario 1
```

### Scenario 2: Fine-Tuning Strategies
Evaluates different unfreezing strategies: `linear_probe`, `last_block`, `selective_20pct`, and `full`.
```bash
python train.py --model resnet50 --scenario 2 --strategy last_block
python train.py --model resnet50 --scenario 2 --strategy selective_20pct
python train.py --model resnet50 --scenario 2 --strategy full
```

### Scenario 3: Few-Shot Learning Analysis
Evaluates data efficiency on subsets of the training data (5%, 20%, 100%).
```bash
python train.py --model resnet50 --scenario 3 --pct 5
python train.py --model resnet50 --scenario 3 --pct 20
python train.py --model resnet50 --scenario 3 --pct 100
```

### Scenario 4: Corruption Robustness Evaluation
Evaluates the best Scenario 1 model (linear probe) under various test-time corruptions (no retraining).
```bash
# Available corruptions: gaussian_0.05, gaussian_0.10, gaussian_0.20, motion_blur, brightness_dark, brightness_bright
python train.py --model resnet50 --scenario 4 --corruption gaussian_0.10
python train.py --model resnet50 --scenario 4 --corruption motion_blur
```

### Scenario 5: Layer-Wise Feature Probing
Extracts features from early, mid, and final layers, freezing the backbone, and trains a linear classifier on them.
```bash
python train.py --model resnet50 --scenario 5 --depth early
python train.py --model resnet50 --scenario 5 --depth mid
python train.py --model resnet50 --scenario 5 --depth final
```

---

## 📊 Aggregating Results
Once the scenarios have been executed, you can automatically aggregate the metrics, plot cross-model comparisons, and generate CSV reports.

```bash
python analyze_results.py
```
This script collects data from all `metrics.json` and `model_info.json` files and outputs summaries in `results/aggregate/`.

---

## 📁 Repository Structure

```text
.
├── train.py                  # Unified entry point for all scenarios
├── analyze_results.py        # Aggregation script for cross-comparison tables & plots
├── requirements.txt          # Python dependencies
├── configs/                  # Additional configuration files if necessary
├── scenarios/                # Individual scenario execution scripts
│   ├── scenario1_linear_probe.py
│   ├── scenario2_finetune.py
│   ├── scenario3_fewshot.py
│   ├── scenario4_corruption.py
│   └── scenario5_layer_probe.py
├── utils/                    # Utility scripts
│   ├── dataset.py            # Dataset loading and splitting
│   ├── metrics.py            # Accuracy, feature norms, confusion matrix
│   ├── model_utils.py        # Loading timm models, freezing strategies, MACs/FLOPs
│   ├── trainer.py            # Shared training/validation loops
│   └── transforms.py         # Transforms and corruptions
└── results/                  # Generated results, plots, and checkpoints (created dynamically)
```

## ⚠️ Notes on Hardware Constraints & Batch Sizes
To accommodate environments with limited VRAM (e.g., 6GB on an RTX 4050) and to maintain absolute fairness during cross-model comparisons, the batch size has been explicitly set to **32** for **all models** (ResNet50, Inception v3, and DenseNet121) across all scenarios. This prevents Out-Of-Memory (OOM) errors during full fine-tuning (Scenario 2, `--strategy full`), which utilizes the most memory, while ensuring all models are evaluated under identical training conditions.
