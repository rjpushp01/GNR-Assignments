# GNR 638 — Assignment 1: Custom Deep Learning Framework

A deep learning framework built from scratch with a C++ computational backend and Python frontend. Implements a complete CNN training and evaluation pipeline **without using any external deep learning or numerical libraries** (beyond standard library + OpenCV for image I/O).

---

## Table of Contents

1. [Quick Start (Full Pipeline)](#quick-start-full-pipeline)
2. [Project Structure](#project-structure)
3. [Prerequisites](#prerequisites)
4. [Setup Instructions](#setup-instructions)
5. [Training a Model](#training-a-model)
6. [Evaluating a Model](#evaluating-a-model)
7. [Generating Plots](#generating-plots)
8. [Running Unit Tests](#running-unit-tests)
9. [Output Files & Where to Find Them](#output-files--where-to-find-them)
10. [Model Architectures](#model-architectures)
11. [Regularization & Overfitting Prevention](#regularization--overfitting-prevention)
12. [Training Results](#training-results)
13. [Metrics Reported](#metrics-reported)

---

## Quick Start (Full Pipeline)

```bash
# 1. Setup
conda env create -f environment.yml
conda activate gnr638_ass
cmake -S . -B build -DPYTHON_EXECUTABLE=$(which python)
cmake --build build

# 2. Train MNIST (fast, ~4 min)
python train.py --dataset mnist --data_path data_1 --epochs 5 --lr 0.001

# 3. Train CIFAR-100 with regularization (~1.5 hrs for 10 epochs)
python train.py --dataset cifar --data_path data_2 --epochs 10 --lr 0.001 --weight_decay 1e-4

# 4. Evaluate
python test.py --dataset mnist --data_path data_1 --model_path results/mnist_model.pkl
python test.py --dataset cifar --data_path data_2 --model_path results/cifar_model.pkl

# 5. Generate plots
python plot_metrics.py --log_file results/training_log_mnist.csv
python plot_metrics.py --log_file results/training_log_cifar.csv
```

All results (model weights, logs, plots) will be in the `results/` directory.

---

## Project Structure

```
Assignment_1/
├── CMakeLists.txt              # Build config for C++ backend
├── environment.yml             # Conda environment specification
├── requirements.txt            # Python dependencies (minimal)
├── train.py                    # Training script (entry point)
├── test.py                     # Evaluation script (entry point)
├── plot_metrics.py             # Plot generation from training logs
│
├── src/                        # C++ backend source code
│   ├── tensor.cpp              # Tensor implementation (CPU)
│   ├── ops.cpp                 # Core ops: matmul, conv2d, relu, dropout, etc.
│   └── bindings.cpp            # pybind11 bindings → my_backend module
│
├── include/                    # C++ headers
│   ├── tensor.hpp
│   └── ops.hpp
│
├── python/                     # Python framework package
│   ├── my_framework/
│   │   ├── __init__.py
│   │   ├── tensor.py           # Tensor wrapper with autograd
│   │   ├── models.py           # MNIST_Model, CIFAR_Model, Dropout, layers
│   │   ├── optim.py            # SGD, Adam (with weight decay / AdamW)
│   │   ├── data.py             # DataLoader (with data augmentation)
│   │   └── model_utils.py      # Parameter/FLOPs/MACs analysis
│   └── my_backend.*.so         # Compiled C++ shared library
│
├── tests/                      # Unit tests
│   ├── test_forward.py
│   └── test_autograd.py
│
├── results/                    # All outputs are saved here
│   ├── training_log_*.csv      # Per-epoch metrics
│   ├── model_summary_*.txt     # Architecture summary
│   ├── *_model.pkl             # Saved model weights
│   ├── test_results.txt        # Test evaluation results
│   ├── combined_dashboard_*.png # Full training dashboard (6 panels)
│   ├── loss_curve_*.png        # Train vs val loss
│   ├── accuracy_curve_*.png    # Train vs val accuracy
│   ├── overfitting_gap_*.png   # Generalization gap analysis
│   ├── lr_schedule_*.png       # Learning rate over epochs
│   ├── memory_usage_*.png      # Peak RSS memory per epoch
│   ├── epoch_time_*.png        # Time per epoch + batch time
│   └── cumulative_time_*.png   # Total wall-clock time
│
├── data_1/                     # MNIST-like dataset (user-provided)
│   ├── class_0/
│   │   ├── img001.png
│   │   └── ...
│   ├── class_1/
│   └── ...
│
└── data_2/                     # CIFAR-like dataset (user-provided)
    ├── class_0/
    ├── class_1/
    └── ...
```

---

## Prerequisites

| Requirement     | Version  | Purpose                                |
|-----------------|----------|----------------------------------------|
| Conda           | any      | Environment management                 |
| Python          | 3.10+    | Frontend language                      |
| C++ Compiler    | GCC 11+  | Backend compilation (C++17)            |
| CMake           | ≥ 3.10   | Build system                           |
| Make            | any      | Build tool                             |
| OpenCV          | 4.x      | Image I/O in C++ backend              |
| pybind11        | ≥ 2.12   | Python ↔ C++ bindings                 |

---

## Setup Instructions

### Step 1 — Create the Conda environment

```bash
conda env create -f environment.yml
conda activate gnr638_ass
```

### Step 2 — Build the C++ backend

```bash
cmake -S . -B build -DPYTHON_EXECUTABLE=$(which python)
cmake --build build
```

This compiles the `my_backend` shared library (`.so` file) and places it in `python/`.

### Step 3 — Verify the build

```bash
python -c "import sys; sys.path.insert(0,'python'); import my_backend; print('Backend OK')"
```

You should see: `Backend OK`

---

## Training a Model

### Command

```bash
python train.py --dataset <DATASET> --data_path <DATA_PATH> \
                --epochs <EPOCHS> --batch_size <BATCH_SIZE> \
                --lr <LEARNING_RATE> --weight_decay <WEIGHT_DECAY>
```

### Parameters

| Parameter        | Required | Default | Description                                                                 |
|------------------|----------|---------|-----------------------------------------------------------------------------|
| `--dataset`      | **Yes**  | —       | `mnist` or `cifar`. Selects the model architecture.                         |
| `--data_path`    | **Yes**  | —       | Path to the dataset root directory (subdirectories per class with `.png` images). |
| `--epochs`       | No       | `5`     | Number of training epochs.                                                  |
| `--batch_size`   | No       | `32`    | Batch size for training and validation.                                     |
| `--lr`           | No       | `0.001` | Learning rate for the Adam optimizer.                                       |
| `--weight_decay` | No       | `0.0`   | AdamW decoupled weight decay (L2 regularization). Recommended: `1e-4`.      |

### Examples

```bash
# Train MNIST model for 10 epochs
python train.py --dataset mnist --data_path data_1 --epochs 10 --lr 0.001

# Train CIFAR model for 20 epochs with regularization
python train.py --dataset cifar --data_path data_2 --epochs 20 --batch_size 64 \
                --lr 0.001 --weight_decay 1e-4
```

### What gets printed during training

1. **Dataset loading time** — how long it takes to read all image paths from disk
2. **Model summary table** — layer-by-layer breakdown of output shapes, parameters, MACs, and FLOPs
3. **Training configuration** — all hyperparameters, sample counts, total params/FLOPs, weight decay
4. **Per-step progress** — loss and accuracy every 10 batches, with batch time
5. **Per-epoch summary** — train loss/acc, val loss/acc, epoch time (train + val breakdown), average batch time, cumulative time, memory usage
6. **End-of-training summary** — total time, best validation accuracy and its epoch, final metrics

### What gets saved

| File                                    | Description                                    |
|-----------------------------------------|------------------------------------------------|
| `results/training_log_<dataset>.csv`    | CSV with per-epoch metrics (11 columns)        |
| `results/<dataset>_model.pkl`           | Saved model weights (pickle)                   |
| `results/model_summary_<dataset>.txt`   | Model summary table with params/MACs/FLOPs     |

### Dataset format

The dataset root directory must follow this structure:

```
<data_path>/
├── class_name_0/
│   ├── image1.png
│   ├── image2.png
│   └── ...
├── class_name_1/
│   ├── image1.png
│   └── ...
└── ...
```

- **MNIST** (`--dataset mnist`): Images should be grayscale PNG files (resized to 32×32 internally).
- **CIFAR** (`--dataset cifar`): Images should be 32×32 RGB PNG files.
- The data is automatically split internally: **70% train**, **20% validation**, **10% test**.

---

## Evaluating a Model

### Command

```bash
python test.py --dataset <DATASET> --data_path <DATA_PATH> \
               --model_path <MODEL_PATH> --batch_size <BATCH_SIZE>
```

### Parameters

| Parameter       | Required | Default | Description                                                                 |
|-----------------|----------|---------|-----------------------------------------------------------------------------|
| `--dataset`     | **Yes**  | —       | `mnist` or `cifar`. Must match the model that was trained.                  |
| `--data_path`   | **Yes**  | —       | Path to the dataset root directory (same format as training).               |
| `--model_path`  | **Yes**  | —       | Path to the saved model weights pickle file (e.g., `results/mnist_model.pkl`). |
| `--batch_size`  | No       | `32`    | Batch size for evaluation.                                                  |

### Examples

```bash
# Evaluate MNIST model
python test.py --dataset mnist --data_path data_1 --model_path results/mnist_model.pkl

# Evaluate CIFAR model
python test.py --dataset cifar --data_path data_2 --model_path results/cifar_model.pkl
```

### What gets printed and saved

- **Model summary** (params, MACs, FLOPs)
- **Dataset loading time**
- **Test accuracy**
- **Evaluation time**
- Results are appended to `results/test_results.txt`

---

## Generating Plots

### Command

```bash
python plot_metrics.py --log_file <LOG_FILE>
```

### Parameters

| Parameter     | Required | Default                              | Description                          |
|---------------|----------|--------------------------------------|--------------------------------------|
| `--log_file`  | No       | `results/training_log_mnist.csv`     | Path to the training log CSV file.   |

### Examples

```bash
# Plot MNIST training metrics
python plot_metrics.py --log_file results/training_log_mnist.csv

# Plot CIFAR training metrics
python plot_metrics.py --log_file results/training_log_cifar.csv
```

### Generated plots (8 per dataset)

| Plot file                                      | Description                                    |
|------------------------------------------------|------------------------------------------------|
| `results/loss_curve_*.png`                     | Train vs. validation loss per epoch            |
| `results/accuracy_curve_*.png`                 | Train vs. validation accuracy per epoch        |
| `results/overfitting_gap_*.png`                | Generalization gap (train − val accuracy)      |
| `results/lr_schedule_*.png`                    | Learning rate schedule across epochs           |
| `results/memory_usage_*.png`                   | Peak RSS memory usage per epoch                |
| `results/epoch_time_*.png`                     | Time per epoch (bars) + avg batch time (line)  |
| `results/cumulative_time_*.png`                | Cumulative wall-clock training time            |
| `results/combined_dashboard_*.png`             | 3×2 combined dashboard with all metrics        |

---

## Running Unit Tests

```bash
python -m unittest discover tests
```

This runs the framework correctness tests in `tests/test_forward.py` and `tests/test_autograd.py`.

---

## Output Files & Where to Find Them

All outputs are saved to the `results/` directory:

| File                                 | Created By      | Description                                    |
|--------------------------------------|-----------------|------------------------------------------------|
| `training_log_<dataset>.csv`         | `train.py`      | Per-epoch training and validation metrics      |
| `model_summary_<dataset>.txt`        | `train.py`      | Model architecture table with params/FLOPs     |
| `<dataset>_model.pkl`                | `train.py`      | Saved model weights                            |
| `test_results.txt`                   | `test.py`       | Evaluation results (appended each run)         |
| `loss_curve_*.png`                   | `plot_metrics`  | Loss curve plot                                |
| `accuracy_curve_*.png`               | `plot_metrics`  | Accuracy curve plot                            |
| `overfitting_gap_*.png`              | `plot_metrics`  | Generalization gap bar chart                   |
| `lr_schedule_*.png`                  | `plot_metrics`  | Learning rate schedule                         |
| `memory_usage_*.png`                 | `plot_metrics`  | Memory usage over epochs                       |
| `epoch_time_*.png`                   | `plot_metrics`  | Per-epoch time + batch time                    |
| `cumulative_time_*.png`              | `plot_metrics`  | Cumulative time plot                           |
| `combined_dashboard_*.png`           | `plot_metrics`  | Combined 3×2 dashboard subplot                 |

---

## Model Architectures

### MNIST Model (LeNet-style)

| Layer     | Type     | Output Shape     | Parameters |
|-----------|----------|------------------|------------|
| conv1     | Conv2d   | [B, 6, 28, 28]  | 156        |
| relu1     | ReLU     | [B, 6, 28, 28]  | 0          |
| pool1     | MaxPool  | [B, 6, 14, 14]  | 0          |
| conv2     | Conv2d   | [B, 16, 10, 10] | 2,416      |
| relu2     | ReLU     | [B, 16, 10, 10] | 0          |
| pool2     | MaxPool  | [B, 16, 5, 5]   | 0          |
| flatten   | Flatten  | [B, 400]         | 0          |
| fc1       | Linear   | [B, 120]         | 48,120     |
| relu3     | ReLU     | [B, 120]         | 0          |
| fc2       | Linear   | [B, 84]          | 10,164     |
| relu4     | ReLU     | [B, 84]          | 0          |
| fc3       | Linear   | [B, 10]          | 850        |
| **Total** |          |                  | **61,706** |

### CIFAR Model (Mini-VGG with Regularization)

| Layer     | Type       | Output Shape     | Parameters |
|-----------|------------|------------------|------------|
| conv1     | Conv2d     | [B, 32, 32, 32] | 896        |
| relu1     | ReLU       | [B, 32, 32, 32] | 0          |
| conv2     | Conv2d     | [B, 32, 32, 32] | 9,248      |
| relu2     | ReLU       | [B, 32, 32, 32] | 0          |
| pool1     | MaxPool2d  | [B, 32, 16, 16] | 0          |
| **drop1** | **Dropout(0.25)** | [B, 32, 16, 16] | **0** |
| conv3     | Conv2d     | [B, 64, 16, 16] | 18,496     |
| relu3     | ReLU       | [B, 64, 16, 16] | 0          |
| conv4     | Conv2d     | [B, 64, 16, 16] | 36,928     |
| relu4     | ReLU       | [B, 64, 16, 16] | 0          |
| pool2     | MaxPool2d  | [B, 64, 8, 8]   | 0          |
| **drop2** | **Dropout(0.25)** | [B, 64, 8, 8] | **0**  |
| flatten   | Flatten    | [B, 4096]        | 0          |
| fc1       | Linear     | [B, 256]         | 1,048,832  |
| relu5     | ReLU       | [B, 256]         | 0          |
| **drop3** | **Dropout(0.5)** | [B, 256]    | **0**      |
| fc2       | Linear     | [B, 100]         | 25,700     |
| **Total** |            |                  | **~1.14M** |

> **Bold** rows indicate regularization layers added to combat overfitting. Dropout layers have zero trainable parameters but are crucial for generalization.

---

## Regularization & Overfitting Prevention

The CIFAR-100 model employs three complementary regularization techniques to reduce the train-val accuracy gap:

### 1. Dropout

Randomly zeroes elements of the input tensor during training with a given probability, forcing the network to learn redundant representations.

| Location                | Dropout Rate | Purpose                                    |
|-------------------------|--------------|--------------------------------------------|
| After conv block 1      | 25%          | Prevent co-adaptation of conv features     |
| After conv block 2      | 25%          | Prevent co-adaptation of conv features     |
| After FC1 (256 units)   | 50%          | Strong regularization in high-param layer  |

- **Implementation**: Custom C++ op (`dropout`, `dropout_backward`) with inverted dropout scaling (output × `1/(1-p)` during training so no scaling needed at test time).
- **Train/Eval mode**: `model.train()` enables dropout; `model.eval()` disables it (pass-through).

### 2. Data Augmentation — Random Horizontal Flip

Each training image is independently flipped horizontally with 50% probability, effectively doubling the dataset diversity.

- **Implementation**: Custom C++ op (`random_horizontal_flip`) applied in the `DataLoader` during training only.
- **No overhead at test time**: Augmentation is disabled when `mode != 'train'`.

### 3. Weight Decay (AdamW)

Decoupled weight decay regularization applied directly to the parameters, penalizing large weights.

```bash
# Enable with --weight_decay flag
python train.py --dataset cifar --data_path data_2 --weight_decay 1e-4
```

- **Implementation**: Custom C++ op (`adam_step_wd`) implementing the AdamW update rule: `θ ← θ - lr × (m̂/(√v̂ + ε) + λ × θ)`.
- **Falls back** to standard Adam when `weight_decay=0.0` (default).

---

## Training Results

### MNIST (5 epochs, lr=0.001)

| Metric              | Value    |
|---------------------|----------|
| Best Val Accuracy   | 96.98%   |
| Final Train Acc     | 97.86%   |
| Final Val Acc       | 96.98%   |
| Generalization Gap  | 0.9%     |
| Total Training Time | 3.6 min  |
| Peak Memory (RSS)   | 223 MB   |

#### MNIST Training Dashboard

![MNIST Training Dashboard](results/combined_dashboard_training_log_mnist.png)

### CIFAR-100 (10 epochs, lr=0.001, no weight decay — baseline)

| Metric              | Value    |
|---------------------|----------|
| Best Val Accuracy   | 35.91%   |
| Final Train Acc     | 86.31%   |
| Final Val Acc       | 33.33%   |
| Generalization Gap  | 53.0%    |
| Total Training Time | 97.1 min |
| Peak Memory (RSS)   | 2,614 MB |

#### CIFAR-100 Training Dashboard

![CIFAR-100 Training Dashboard](results/combined_dashboard_training_log_cifar.png)

> **Note**: The CIFAR-100 results above are from an earlier training run **without** regularization (no Dropout, no augmentation, no weight decay). The newly added regularization layers (Dropout, random horizontal flip, AdamW weight decay) are expected to significantly reduce the generalization gap when re-trained.

### Individual Plots

<details>
<summary>📊 MNIST — Individual Metric Plots</summary>

| Loss Curve | Accuracy Curve |
|:---:|:---:|
| ![](results/loss_curve_training_log_mnist.png) | ![](results/accuracy_curve_training_log_mnist.png) |

| Overfitting Gap | Learning Rate |
|:---:|:---:|
| ![](results/overfitting_gap_training_log_mnist.png) | ![](results/lr_schedule_training_log_mnist.png) |

| Epoch Time | Memory Usage |
|:---:|:---:|
| ![](results/epoch_time_training_log_mnist.png) | ![](results/memory_usage_training_log_mnist.png) |

| Cumulative Time |
|:---:|
| ![](results/cumulative_time_training_log_mnist.png) |

</details>

<details>
<summary>📊 CIFAR-100 — Individual Metric Plots</summary>

| Loss Curve | Accuracy Curve |
|:---:|:---:|
| ![](results/loss_curve_training_log_cifar.png) | ![](results/accuracy_curve_training_log_cifar.png) |

| Overfitting Gap | Learning Rate |
|:---:|:---:|
| ![](results/overfitting_gap_training_log_cifar.png) | ![](results/lr_schedule_training_log_cifar.png) |

| Epoch Time | Memory Usage |
|:---:|:---:|
| ![](results/epoch_time_training_log_cifar.png) | ![](results/memory_usage_training_log_cifar.png) |

| Cumulative Time |
|:---:|
| ![](results/cumulative_time_training_log_cifar.png) |

</details>

---

## Metrics Reported

During training and evaluation, the following metrics are computed and reported:

| Metric                         | Printed | Saved to File         |
|--------------------------------|---------|-----------------------|
| Model architecture summary     | ✅      | `model_summary_*.txt` |
| Total trainable parameters     | ✅      | `model_summary_*.txt` |
| MACs per forward pass          | ✅      | `model_summary_*.txt` |
| FLOPs per forward pass         | ✅      | `model_summary_*.txt` |
| Dataset loading time (seconds) | ✅      | Console output        |
| Training loss per epoch        | ✅      | `training_log_*.csv`  |
| Training accuracy per epoch    | ✅      | `training_log_*.csv`  |
| Validation loss per epoch      | ✅      | `training_log_*.csv`  |
| Validation accuracy per epoch  | ✅      | `training_log_*.csv`  |
| Epoch time (seconds)           | ✅      | `training_log_*.csv`  |
| Cumulative training time       | ✅      | `training_log_*.csv`  |
| Average batch time             | ✅      | `training_log_*.csv`  |
| Memory usage (RSS MB)          | ✅      | `training_log_*.csv`  |
| Learning rate per epoch        | ✅      | `training_log_*.csv`  |
| Generalization gap             | ✅      | Plots (computed)      |
| Test accuracy                  | ✅      | `test_results.txt`    |
| Evaluation time                | ✅      | `test_results.txt`    |

