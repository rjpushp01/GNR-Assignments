"""
Scenario 1: Linear Probe Transfer
==================================
Freeze all backbone parameters. Train only the linear classifier head.
Reports: train/val accuracy curves, confusion matrix, PCA/t-SNE embeddings.

Usage:
    python train.py --model resnet50 --scenario 1
    python train.py --model inception_v3 --scenario 1
    python train.py --model densenet121 --scenario 1
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn

from utils.dataset import get_dataloaders
from utils.model_utils import load_model, freeze_backbone, get_model_info, count_params
from utils.trainer import run_training, validate
from utils.visualization import (
    plot_accuracy_curves, plot_loss_curves, plot_confusion_matrix,
    plot_embeddings, save_metrics_json,
)
from utils.model_utils import MODEL_CONFIGS


def run(model_name: str, data_root: str, save_dir: str, device: torch.device):
    """Run Scenario 1: Linear Probe Transfer."""
    print(f"\n{'='*60}")
    print(f"  SCENARIO 1: Linear Probe — {model_name.upper()}")
    print(f"{'='*60}")
    os.makedirs(save_dir, exist_ok=True)

    cfg = MODEL_CONFIGS[model_name]

    # ── Load data ──────────────────────────────────────────────────────────
    train_loader, val_loader, class_names = get_dataloaders(
        model_name=model_name,
        data_root=data_root,
        pct=1.0,
        seed=42,
    )
    print(f"  Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}")
    print(f"  Classes: {len(class_names)}")

    # ── Load & freeze model ────────────────────────────────────────────────
    model = load_model(model_name, num_classes=len(class_names), pretrained=True).to(device)
    freeze_backbone(model)

    # Print & save model efficiency info
    model_info = get_model_info(model, model_name, save_dir)

    param_info = count_params(model)
    print(f"  [Linear Probe] Trainable: {param_info['trainable_params']:,} / {param_info['total_params']:,} "
          f"({param_info['trainable_pct']:.2f}%)")

    # ── Train ──────────────────────────────────────────────────────────────
    history = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=cfg["epochs_full"],
        lr=cfg["lr"],
        save_dir=save_dir,
        num_classes=len(class_names),
    )

    # ── Plots: accuracy & loss curves ─────────────────────────────────────
    plot_accuracy_curves(
        history["train_acc"], history["val_acc"],
        save_path=os.path.join(save_dir, "train_val_accuracy.png"),
        title=f"Linear Probe — {model_name} — Accuracy",
    )
    plot_loss_curves(
        history["train_loss"], history["val_loss"],
        save_path=os.path.join(save_dir, "train_val_loss.png"),
        title=f"Linear Probe — {model_name} — Loss",
    )

    # ── Confusion matrix ───────────────────────────────────────────────────
    # Reload best weights
    model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pth"), map_location=device))
    criterion = nn.CrossEntropyLoss()
    _, best_val_acc, cm = validate(
        model, val_loader, criterion, device,
        num_classes=len(class_names), return_cm=True,
    )
    plot_confusion_matrix(
        cm, class_names,
        save_path=os.path.join(save_dir, "confusion_matrix.png"),
        title=f"Linear Probe CM — {model_name}",
    )

    # ── Feature embeddings ─────────────────────────────────────────────────
    features, labels_arr = _extract_features(model, val_loader, device)
    for method in ["pca", "tsne"]:
        plot_embeddings(
            features, labels_arr, class_names,
            save_path=os.path.join(save_dir, f"embeddings_{method}.png"),
            method=method,
            title=f"Linear Probe — {model_name} — {method.upper()}",
        )

    # ── Save summary metrics ───────────────────────────────────────────────
    metrics = {
        "scenario": "linear_probe",
        "model": model_name,
        "best_val_acc": history["best_val_acc"],
        "final_train_acc": history["train_acc"][-1],
        "final_val_acc": history["val_acc"][-1],
        "trainable_pct": param_info["trainable_pct"],
        **{k: model_info[k] for k in ["total_params", "trainable_params", "macs_str", "flops_str"] if k in model_info},
    }
    save_metrics_json(metrics, os.path.join(save_dir, "metrics.json"))
    print(f"\n  ✓ Best Val Accuracy: {history['best_val_acc']:.2f}%")
    print(f"  ✓ Results saved to: {save_dir}")


@torch.no_grad()
def _extract_features(model, loader, device):
    """Extract penultimate layer features (before classifier) from the model."""
    model.eval()
    all_feats, all_labels = [], []

    # Register hook on the global avg pool output (before classifier)
    hooked = []
    def hook_fn(module, inp, out):
        if out.dim() > 2:
            hooked.append(out.mean(dim=[2, 3]).cpu())
        else:
            hooked.append(out.cpu())

    # Use global_pool layer from timm
    handle = None
    if hasattr(model, "global_pool"):
        handle = model.global_pool.register_forward_hook(hook_fn)
    elif hasattr(model, "avgpool"):
        handle = model.avgpool.register_forward_hook(hook_fn)

    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        hooked.clear()
        _ = model(inputs)
        if hooked:
            all_feats.append(hooked[0])
        all_labels.append(labels)

    if handle:
        handle.remove()

    if all_feats:
        features = torch.cat(all_feats, dim=0).numpy()
        # Flatten if needed
        if features.ndim > 2:
            features = features.reshape(features.shape[0], -1)
    else:
        features = np.zeros((len(loader.dataset), 1))

    labels_arr = torch.cat(all_labels, dim=0).numpy()
    return features, labels_arr
