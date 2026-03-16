"""
Scenario 4: Corruption Robustness Evaluation
==============================================
Evaluate model robustness under controlled corruptions at inference time.
No retraining — uses the best linear-probe model from Scenario 1.
Each corruption type is a SEPARATE run.

Corruption types:
  gaussian_0.05   — Gaussian noise σ=0.05
  gaussian_0.10   — Gaussian noise σ=0.10
  gaussian_0.20   — Gaussian noise σ=0.20
  motion_blur     — Gaussian blur simulating motion blur (kernel=15)
  brightness_dark — Brightness reduced to 50%
  brightness_bright — Brightness increased by 50%

Usage:
    python train.py --model resnet50 --scenario 4 --corruption gaussian_0.05
    python train.py --model resnet50 --scenario 4 --corruption gaussian_0.10
    python train.py --model resnet50 --scenario 4 --corruption gaussian_0.20
    python train.py --model resnet50 --scenario 4 --corruption motion_blur
    python train.py --model resnet50 --scenario 4 --corruption brightness_dark
    python train.py --model resnet50 --scenario 4 --corruption brightness_bright
"""

import os
import json
import time
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset

from utils.model_utils import load_model, freeze_backbone, MODEL_CONFIGS
from utils.transforms import get_corruption_transform
from utils.dataset import stratified_split
from utils.metrics import compute_accuracy, update_confusion_matrix, corruption_error, relative_robustness
from utils.visualization import save_metrics_json, plot_confusion_matrix
from tqdm import tqdm

VALID_CORRUPTIONS = [
    "gaussian_0.05", "gaussian_0.10", "gaussian_0.20",
    "motion_blur", "brightness_dark", "brightness_bright",
]

CORRUPTION_PARAMS = {
    "gaussian_0.05": ("gaussian", 0.05),
    "gaussian_0.10": ("gaussian", 0.10),
    "gaussian_0.20": ("gaussian", 0.20),
    "motion_blur":   ("motion_blur", None),
    "brightness_dark":  ("brightness_dark", None),
    "brightness_bright": ("brightness_bright", None),
}


@torch.no_grad()
def evaluate_corruption(model, loader, device, num_classes=30):
    model.eval()
    total_acc = 0.0
    cm = None

    for inputs, labels in tqdm(loader, desc="  Eval", leave=False):
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        outputs = model(inputs)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        total_acc += compute_accuracy(outputs, labels) * inputs.size(0)

    n = len(loader.dataset)
    return total_acc / n


def run(
    model_name: str,
    data_root: str,
    save_dir: str,
    device: torch.device,
    corruption: str = "gaussian_0.10",
    model_checkpoint_dir: str = None,
):
    """
    Evaluate a pretrained linear probe model under a specific corruption.

    Args:
        model_checkpoint_dir: Directory of the Scenario 1 best_model.pth.
                              If None, inferred from results/{model_name}/scenario1_linear_probe/
    """
    start_time = time.time() # Start total execution timer
    assert corruption in VALID_CORRUPTIONS, f"corruption must be one of {VALID_CORRUPTIONS}"

    print(f"\n{'='*60}")
    print(f"  SCENARIO 4: Corruption [{corruption}] — {model_name.upper()}")
    print(f"{'='*60}")
    os.makedirs(save_dir, exist_ok=True)

    cfg = MODEL_CONFIGS[model_name]
    num_classes = 30

    # ── Load model from Scenario 1 checkpoint ─────────────────────────────
    if model_checkpoint_dir is None:
        # Default: look relative to results/
        base = os.path.dirname(os.path.dirname(save_dir))
        model_checkpoint_dir = os.path.join(base, "scenario1_linear_probe")
    
    checkpoint_path = os.path.join(model_checkpoint_dir, "best_model.pth")

    model = load_model(model_name, num_classes=num_classes, pretrained=True).to(device)
    freeze_backbone(model)

    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"  Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"  ⚠ Checkpoint not found at {checkpoint_path}. Using pretrained weights.")

    t_ds = time.time()
    # ── Build corrupted validation loader ─────────────────────────────────
    corruption_type, severity = CORRUPTION_PARAMS[corruption]
    corrupt_transform = get_corruption_transform(model_name, corruption_type, severity)

    # Load with full dataset, use same val split as training (seed=42, val_ratio=0.3)
    full_dataset = ImageFolder(root=data_root, transform=corrupt_transform)
    class_names = full_dataset.classes
    _, val_indices = stratified_split(full_dataset, val_ratio=0.3, seed=42)
    val_subset = Subset(full_dataset, val_indices)
    
    val_loader = DataLoader(
        val_subset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    print(f"  Corrupted val samples: {len(val_subset)}")

    # ── Evaluate clean baseline too (for relative robustness) ─────────────
    from utils.transforms import get_transforms
    _, clean_transform = get_transforms(model_name)
    clean_dataset = ImageFolder(root=data_root, transform=clean_transform)
    clean_val_subset = Subset(clean_dataset, val_indices)
    clean_loader = DataLoader(
        clean_val_subset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    dataset_setup_time = time.time() - t_ds
    print(f"  Dataset setup took: {dataset_setup_time:.2f}s")
    
    t0 = time.time()
    clean_acc = evaluate_corruption(model, clean_loader, device, num_classes)
    corrupted_acc = evaluate_corruption(model, val_loader, device, num_classes)
    eval_time = time.time() - t0

    ce = corruption_error(corrupted_acc)
    rr = relative_robustness(corrupted_acc, clean_acc)

    print(f"\n  Clean accuracy:     {clean_acc:.2f}%")
    print(f"  Corrupted accuracy: {corrupted_acc:.2f}%")
    print(f"  Corruption Error:   {ce:.4f}")
    print(f"  Relative Robustness:{rr:.4f}")

    # ── Save metrics ───────────────────────────────────────────────────────
    metrics = {
        "scenario": "corruption_robustness",
        "model": model_name,
        "corruption": corruption,
        "corruption_type": corruption_type,
        "severity": severity,
        "clean_acc": clean_acc,
        "corrupted_acc": corrupted_acc,
        "corruption_error": ce,
        "relative_robustness": rr,
        "total_eval_time": eval_time,
    }
    save_metrics_json(metrics, os.path.join(save_dir, "metrics.json"))
    print(f"  ✓ Results saved to: {save_dir}")
    return metrics
