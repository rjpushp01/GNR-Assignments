# Results Directory Structure
## GNR 638 Assignment 2 — AID Aerial Image Classification

This directory contains all experiment outputs, organized by **model → scenario → sub-scenario**.

---

## Directory Layout

```
results/
├── aggregate/                          ← Cross-model comparison plots & CSVs
│   ├── model_efficiency_detailed.csv
│   ├── fewshot_detailed.csv
│   ├── finetune_detailed.csv
│   ├── corruption_detailed.csv
│   ├── layer_probe_detailed.csv
│   ├── scenario1_detailed.csv
│   ├── fewshot_comparison.png
│   ├── finetune_comparison.png
│   ├── layer_probe_comparison.png
│   ├── training_dynamics_comparison.png
│   ├── finetune_time_breakdown.png
│   ├── fewshot_time_breakdown.png
│   └── corruption_bar_chart.png
│
├── resnet50/
│   ├── scenario1_linear_probe/
│   │   ├── best_model.pth              ← Best checkpoint
│   │   ├── history.json                ← Full training history
│   │   ├── model_info.json             ← Params, MACs, FLOPs
│   │   ├── metrics.json                ← Summary metrics
│   │   ├── train_val_accuracy.png
│   │   ├── train_val_loss.png
│   │   ├── confusion_matrix.png
│   │   ├── embeddings_pca.png
│   │   └── embeddings_tsne.png
│   │
│   ├── scenario2_finetuning/
│   │   ├── linear_probe/               ← Same structure as above
│   │   ├── last_block/
│   │   ├── full/
│   │   └── selective_20pct/
│   │
│   ├── scenario3_few_shot/
│   │   ├── pct_100/
│   │   ├── pct_20/
│   │   └── pct_5/
│   │
│   ├── scenario4_corruption/
│   │   ├── gaussian_0.05/              ← metrics.json with CE, robustness
│   │   ├── gaussian_0.10/
│   │   ├── gaussian_0.20/
│   │   ├── motion_blur/
│   │   ├── brightness_dark/
│   │   └── brightness_bright/
│   │
│   └── scenario5_layer_probe/
│       ├── early/                      ← Features + linear classifier
│       ├── mid/
│       └── final/
│
├── inception_v3/                       ← Same structure
└── densenet121/                        ← Same structure
```

---

## Files Per Sub-Scenario

| File | Description |
|------|-------------|
| `best_model.pth` | Best checkpoint (highest val accuracy) |
| `history.json` | Per-epoch train/val loss and accuracy |
| `model_info.json` | Total params, trainable params, MACs, FLOPs |
| `metrics.json` | Summary metrics for the scenario |
| `train_val_accuracy.png` | Accuracy learning curves |
| `train_val_loss.png` | Loss convergence curves |
| `confusion_matrix.png` | Normalized confusion matrix (Scenarios 1, 2) |
| `embeddings_pca.png` | PCA 2D feature embedding (Scenario 1) |
| `embeddings_tsne.png` | t-SNE feature embedding (Scenario 1) |
| `gradient_norms.png` | Layer-wise gradient norms (Scenario 2) |
| `pca_embedding.png` | Fixed-subset PCA plot (Scenario 5) |

---

## Key Metrics
These metrics exist across all `metrics.json` outputs for all scenario checkpoints:

### Hardware Profiling
- **dataset_setup_time**: Overhead initializing `ImageFolder` and splitting stratified subsets.
- **data_fetch_time**: Dataloader CPU loading time bounds.
- **train_pass_time**: Total backward calculation GPU compute time.
- **val_pass_time**: Total independent inference passes calculating Loss limits.
- **avg_epoch_time**: Iteration cycle efficiency scalar.

### Model Efficiency (all scenarios report this)
- **Total Parameters**: Full model parameter count
- **Trainable Parameters**: Parameters updated during training
- **Frozen Parameters**: Architecture identity mapping bounds safely locked relative to active vectors.
- **MACs** (Multiply-Accumulate Operations): Computational complexity
- **FLOPs** = 2 × MACs

### Scenario-Specific Metrics
- **S3 (Few-Shot)**: `relative_drop Δ = (Acc₁₀₀% - Acc₅%) / Acc₁₀₀%`
- **S4 (Corruption)**: `Corruption Error = 1 - Acc_corrupted`, `Relative Robustness = Acc_corrupted / Acc_clean`
- **S5 (Layer Probe)**: Feature norm statistics (`mean_norm`, `std_norm`) mapped globally against dimensional scaling.

---

## How to Re-run

```bash
# Single sub-scenario
python train.py --model resnet50 --scenario 1
python train.py --model resnet50 --scenario 3 --pct 5

# Aggregate cross-model comparisons
python analyze_results.py
```
