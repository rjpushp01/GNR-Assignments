"""
Scenario 3: Few-Shot Learning Analysis
=========================================
Evaluate data efficiency under limited supervision.
Each data regime (5%, 20%, 100%) is a SEPARATE run.

Usage:
    python train.py --model resnet50 --scenario 3 --pct 100
    python train.py --model resnet50 --scenario 3 --pct 20
    python train.py --model resnet50 --scenario 3 --pct 5
"""

import os
import json
import torch
import torch.nn as nn

from utils.dataset import get_dataloaders
from utils.model_utils import load_model, freeze_backbone, get_model_info, count_params, MODEL_CONFIGS
from utils.trainer import run_training
from utils.visualization import plot_accuracy_curves, plot_loss_curves, save_metrics_json

VALID_PCTS = [5, 20, 100]


def run(model_name: str, data_root: str, save_dir: str, device: torch.device, pct: int = 100):
    """
    Run one data regime for Scenario 3.
    pct: 5, 20, or 100 (percentage of training data).
    Applies Linear Probe strategy (frozen backbone, linear head) for fair comparison across data regimes.
    """
    assert pct in VALID_PCTS, f"pct must be one of {VALID_PCTS}"
    pct_float = pct / 100.0

    print(f"\n{'='*60}")
    print(f"  SCENARIO 3: Few-Shot [{pct}% data] — {model_name.upper()}")
    print(f"{'='*60}")
    os.makedirs(save_dir, exist_ok=True)

    cfg = MODEL_CONFIGS[model_name]
    num_epochs = cfg["epochs_full"] if pct == 100 else cfg["epochs_fewshot"]

    # ── Load data ──────────────────────────────────────────────────────────
    train_loader, val_loader, class_names = get_dataloaders(
        model_name=model_name,
        data_root=data_root,
        pct=pct_float,
        seed=42,
    )
    print(f"  Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}")

    # ── Load model (linear probe for consistent comparison) ────────────────
    model = load_model(model_name, num_classes=len(class_names), pretrained=True).to(device)
    freeze_backbone(model)
    model_info = get_model_info(model, model_name, save_dir)
    param_info = count_params(model)

    # ── Train ──────────────────────────────────────────────────────────────
    history = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=num_epochs,
        lr=cfg["lr"],
        save_dir=save_dir,
        num_classes=len(class_names),
    )

    # ── Plots ──────────────────────────────────────────────────────────────
    plot_accuracy_curves(
        history["train_acc"], history["val_acc"],
        save_path=os.path.join(save_dir, "train_val_accuracy.png"),
        title=f"Few-Shot [{pct}% data] — {model_name}",
    )
    plot_loss_curves(
        history["train_loss"], history["val_loss"],
        save_path=os.path.join(save_dir, "train_val_loss.png"),
        title=f"Loss [{pct}% data] — {model_name}",
    )

    # Train-val gap
    final_train_acc = history["train_acc"][-1]
    final_val_acc = history["val_acc"][-1]
    train_val_gap = final_train_acc - final_val_acc

    # ── Save metrics ───────────────────────────────────────────────────────
    metrics = {
        "scenario": "few_shot",
        "model": model_name,
        "pct": pct,
        "num_train_samples": len(train_loader.dataset),
        "num_val_samples": len(val_loader.dataset),
        "best_val_acc": history["best_val_acc"],
        "final_train_acc": final_train_acc,
        "final_val_acc": final_val_acc,
        "train_val_gap": train_val_gap,
        "trainable_pct": param_info["trainable_pct"],
        **{k: model_info[k] for k in ["total_params", "macs_str", "flops_str"] if k in model_info},
    }
    save_metrics_json(metrics, os.path.join(save_dir, "metrics.json"))

    print(f"\n  ✓ Data regime: {pct}%")
    print(f"  ✓ Best Val Accuracy: {history['best_val_acc']:.2f}%")
    print(f"  ✓ Train-Val Gap: {train_val_gap:.2f}%")
    print(f"  ✓ Results saved to: {save_dir}")
    return metrics
