"""
Scenario 2: Fine-Tuning Strategies
=====================================
Compare 4 fine-tuning strategies for a given model.
Each strategy is a SEPARATE run (separate sub-directory).

Strategies:
  linear_probe   — freeze all, train classifier only
  last_block     — unfreeze last conv block + classifier
  full           — unfreeze entire model  
  selective_20pct — unfreeze deepest layers up to 20% of backbone params

Usage:
    python train.py --model resnet50 --scenario 2 --strategy linear_probe
    python train.py --model resnet50 --scenario 2 --strategy last_block
    python train.py --model resnet50 --scenario 2 --strategy full
    python train.py --model resnet50 --scenario 2 --strategy selective_20pct
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn

from utils.dataset import get_dataloaders
from utils.model_utils import (load_model, freeze_backbone, unfreeze_last_block,
                                unfreeze_full, selective_unfreeze, get_model_info, count_params, MODEL_CONFIGS)
from utils.trainer import run_training, validate
from utils.visualization import (
    plot_accuracy_curves, plot_loss_curves, plot_gradient_norms,
    save_metrics_json,
)

VALID_STRATEGIES = ["linear_probe", "last_block", "full", "selective_20pct"]


def apply_strategy(model, model_name: str, strategy: str):
    """Apply the appropriate freeze/unfreeze strategy and return extra info."""
    extra = {}
    if strategy == "linear_probe":
        freeze_backbone(model)
        extra["description"] = "Freeze all backbone layers; train classifier head only."
    elif strategy == "last_block":
        unfreeze_last_block(model, model_name)
        extra["description"] = "Freeze early layers; unfreeze last conv block + classifier."
    elif strategy == "full":
        unfreeze_full(model)
        extra["description"] = "Full fine-tuning of entire backbone + classifier."
    elif strategy == "selective_20pct":
        extra = selective_unfreeze(model, target_pct=0.20)
        extra["description"] = (
            "Unfreeze deepest backbone layers until 20% of backbone params are trainable. "
            "Rationale: deepest layers encode most task-specific features, so updating them "
            "maximises transfer while minimising overfitting risk."
        )
    return extra


def run(model_name: str, data_root: str, save_dir: str, device: torch.device, strategy: str = "full"):
    """Run one fine-tuning strategy for Scenario 2."""
    assert strategy in VALID_STRATEGIES, f"Strategy must be one of {VALID_STRATEGIES}"

    print(f"\n{'='*60}")
    print(f"  SCENARIO 2: Fine-Tuning [{strategy}] — {model_name.upper()}")
    print(f"{'='*60}")
    os.makedirs(save_dir, exist_ok=True)

    cfg = MODEL_CONFIGS[model_name]

    import time
    t_dataset_start = time.time()
    train_loader, val_loader, class_names = get_dataloaders(
        model_name=model_name,
        data_root=data_root,
        pct=1.0,
        seed=42,
    )
    dataset_setup_time = time.time() - t_dataset_start
    print(f"  Dataset setup took: {dataset_setup_time:.2f}s")

    # ── Load & configure model ─────────────────────────────────────────────
    model = load_model(model_name, num_classes=len(class_names), pretrained=True).to(device)
    strategy_info = apply_strategy(model, model_name, strategy)
    param_info = count_params(model)

    model_info = get_model_info(model, model_name, save_dir)
    print(f"  [{strategy}] Trainable: {param_info['trainable_params']:,} / "
          f"{param_info['total_params']:,} ({param_info['trainable_pct']:.2f}%)")

    # Use lower LR for full fine-tune to avoid catastrophic forgetting
    lr = cfg["lr"]
    if strategy in ["full", "last_block"]:
        lr = cfg["lr"] * 0.1  # 1e-4

    # ── Train (with gradient norm collection) ─────────────────────────────
    collect_grads = strategy in ["full", "last_block", "selective_20pct"]
    history = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=cfg["epochs_full"],
        lr=lr,
        save_dir=save_dir,
        collect_grad_norms=collect_grads,
        num_classes=len(class_names),
    )

    # ── Plots ──────────────────────────────────────────────────────────────
    plot_accuracy_curves(
        history["train_acc"], history["val_acc"],
        save_path=os.path.join(save_dir, "train_val_accuracy.png"),
        title=f"Fine-Tune [{strategy}] — {model_name}",
    )
    plot_loss_curves(
        history["train_loss"], history["val_loss"],
        save_path=os.path.join(save_dir, "train_val_loss.png"),
        title=f"Loss Convergence [{strategy}] — {model_name}",
    )

    if collect_grads and "mean_grad_norms" in history:
        plot_gradient_norms(
            history["mean_grad_norms"],
            save_path=os.path.join(save_dir, "gradient_norms.png"),
            title=f"Mean Gradient Norms [{strategy}] — {model_name}",
        )

    # ── Save metrics ───────────────────────────────────────────────────────
    metrics = {
        "scenario": "finetuning",
        "model": model_name,
        "strategy": strategy,
        "best_val_acc": history["best_val_acc"],
        "final_train_acc": history["train_acc"][-1],
        "final_val_acc": history["val_acc"][-1],
        "trainable_params": param_info["trainable_params"],
        "trainable_pct": param_info["trainable_pct"],
        "lr_used": lr,
        "strategy_info": strategy_info,
        "total_training_time": history.get("total_time"),
        "train_pass_time": history.get("total_train_pass_time"),
        "val_pass_time": history.get("total_val_pass_time"),
        "data_fetch_time": history.get("total_data_fetch_time"),
        "dataset_setup_time": dataset_setup_time,
        "avg_epoch_time": history.get("avg_epoch_time"),
        **{k: model_info[k] for k in ["total_params", "macs_str", "flops_str"] if k in model_info},
    }
    save_metrics_json(metrics, os.path.join(save_dir, "metrics.json"))

    print(f"\n  ✓ Strategy: {strategy}")
    print(f"  ✓ Trainable: {param_info['trainable_pct']:.2f}%")
    print(f"  ✓ Best Val Accuracy: {history['best_val_acc']:.2f}%")
    print(f"  ✓ Results saved to: {save_dir}")
    return metrics
