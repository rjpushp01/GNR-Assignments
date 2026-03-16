"""
Scenario 5: Layer-Wise Feature Probing
=========================================
Extract intermediate representations from early, middle, and final layers.
Train a separate linear classifier on features from each depth.
Each depth level is a SEPARATE run.

Layer selection (documented):
  ResNet50:     early=layer1, mid=layer2, final=layer3
  InceptionV3:  early=Mixed_5c, mid=Mixed_6e, final=Mixed_7c
  DenseNet121:  early=denseblock1, mid=denseblock2, final=denseblock4

Usage:
    python train.py --model resnet50 --scenario 5 --depth early
    python train.py --model resnet50 --scenario 5 --depth mid
    python train.py --model resnet50 --scenario 5 --depth final
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.dataset import get_dataloaders, get_fixed_probe_subset
from utils.model_utils import load_model, freeze_backbone, MODEL_CONFIGS, LAYER_HOOKS, register_hook
from utils.metrics import compute_accuracy, compute_feature_norms
from utils.visualization import (
    plot_accuracy_curves, plot_loss_curves, plot_embeddings,
    save_metrics_json,
)
from tqdm import tqdm

VALID_DEPTHS = ["early", "mid", "final"]


@torch.no_grad()
def extract_layer_features(model, loader, model_name: str, depth: str, device: torch.device):
    """
    Use a forward hook to extract features at a specific layer depth.
    Returns (features, labels) as numpy arrays.
    """
    model.eval()
    handle, feature_cache = register_hook(model, model_name, depth)

    all_features, all_labels = [], []

    for inputs, labels in tqdm(loader, desc=f"  Extract [{depth}]", leave=False):
        inputs = inputs.to(device, non_blocking=True)
        feature_cache.clear()
        _ = model(inputs)
        if feature_cache:
            feat = feature_cache[0]  # (B, C)
            if feat.dim() > 2:
                feat = feat.reshape(feat.shape[0], -1)
            all_features.append(feat.numpy())
        all_labels.append(labels.numpy())

    handle.remove()

    features = np.concatenate(all_features, axis=0) if all_features else np.zeros((len(loader.dataset), 1))
    labels = np.concatenate(all_labels, axis=0)
    return features, labels


def train_linear_on_features(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    num_classes: int = 30,
    num_epochs: int = 30,
    lr: float = 1e-2,
    device: torch.device = None,
):
    """Train a simple linear classifier on extracted features."""
    if device is None:
        device = torch.device("cpu")

    X_train = torch.tensor(train_features, dtype=torch.float32).to(device)
    y_train = torch.tensor(train_labels, dtype=torch.long).to(device)
    X_val = torch.tensor(val_features, dtype=torch.float32).to(device)
    y_val = torch.tensor(val_labels, dtype=torch.long).to(device)

    # Normalise features
    mean = X_train.mean(0, keepdim=True)
    std = X_train.std(0, keepdim=True) + 1e-8
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    in_dim = X_train.shape[1]
    classifier = nn.Linear(in_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    train_accs, val_accs, train_losses, val_losses = [], [], [], []
    best_val_acc = 0.0

    # Mini-batch training
    batch_size = 256
    n = X_train.shape[0]

    import time
    t_start = time.time()
    epoch_times = []

    for epoch in range(num_epochs):
        t0 = time.time()
        classifier.train()
        perm = torch.randperm(n)
        epoch_loss, epoch_acc = 0.0, 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_train[idx], y_train[idx]
            out = classifier(xb)
            loss = criterion(out, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
            epoch_acc += compute_accuracy(out.detach(), yb) * xb.size(0)
        train_accs.append(epoch_acc / n)
        train_losses.append(epoch_loss / n)

        classifier.eval()
        with torch.no_grad():
            out_val = classifier(X_val)
            val_loss = criterion(out_val, y_val).item()
            val_acc = compute_accuracy(out_val, y_val)
            val_accs.append(val_acc)
            val_losses.append(val_loss)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
        epoch_times.append(time.time() - t0)
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == num_epochs - 1:
            print(f"    Epoch {epoch+1:2d}/{num_epochs} | train_loss={train_losses[-1]:.4f}  "
                  f"train_acc={train_accs[-1]:.2f}%  val_acc={val_accs[-1]:.2f}%")

    total_train_time = time.time() - t_start
    return {
        "train_accs": train_accs,
        "val_accs": val_accs,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_acc": best_val_acc,
        "total_train_time": total_train_time,
        "avg_epoch_time": np.mean(epoch_times),
    }


def run(
    model_name: str,
    data_root: str,
    save_dir: str,
    device: torch.device,
    depth: str = "final",
):
    """
    Run one depth level for Scenario 5.
    depth: 'early', 'mid', or 'final'
    """
    import time
    start_total = time.time()
    assert depth in VALID_DEPTHS, f"depth must be one of {VALID_DEPTHS}"

    print(f"\n{'='*60}")
    print(f"  SCENARIO 5: Layer Probe [{depth}] — {model_name.upper()}")
    print(f"  Layer: {LAYER_HOOKS[model_name][depth]}")
    print(f"{'='*60}")
    os.makedirs(save_dir, exist_ok=True)

    cfg = MODEL_CONFIGS[model_name]
    num_classes = 30

    # ── Load frozen backbone ────────────────────────────────────────────────
    model = load_model(model_name, num_classes=num_classes, pretrained=True).to(device)
    # Freeze everything (we are feature extractors only)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    # ── Full train/val data for classifier training ────────────────────────
    import time
    from utils.transforms import get_transforms
    from torchvision.datasets import ImageFolder
    from torch.utils.data import Subset
    from utils.dataset import stratified_split

    t_ds = time.time()
    _, val_transform = get_transforms(model_name)
    full_dataset = ImageFolder(root=data_root, transform=val_transform)
    class_names = full_dataset.classes
    train_indices, val_indices = stratified_split(full_dataset, val_ratio=0.3, seed=42)

    train_loader = DataLoader(
        Subset(full_dataset, train_indices),
        batch_size=cfg["batch_size"], shuffle=False, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(full_dataset, val_indices),
        batch_size=cfg["batch_size"], shuffle=False, num_workers=4, pin_memory=True,
    )
    dataset_setup_time = time.time() - t_ds
    print(f"  Dataset setup took: {dataset_setup_time:.2f}s")

    # ── Extract features ────────────────────────────────────────────────────
    print(f"  Extracting {depth} features from '{LAYER_HOOKS[model_name][depth]}'...")
    t_ext = time.time()
    train_features, train_labels = extract_layer_features(model, train_loader, model_name, depth, device)
    val_features, val_labels = extract_layer_features(model, val_loader, model_name, depth, device)
    extraction_time = time.time() - t_ext
    print(f"  Feature dim: {train_features.shape[1]}")

    # Feature norm statistics
    feat_norms = compute_feature_norms(val_features)
    print(f"  Feature norms — mean: {feat_norms['mean_norm']:.3f}, std: {feat_norms['std_norm']:.3f}")

    # ── Train linear classifier on features ────────────────────────────────
    history = train_linear_on_features(
        train_features, train_labels,
        val_features, val_labels,
        num_classes=num_classes,
        num_epochs=50,
        lr=1e-2,
        device=device,
    )

    # ── Plots ────────────────────────────────────────────────────────────────
    plot_accuracy_curves(
        history["train_accs"], history["val_accs"],
        save_path=os.path.join(save_dir, "train_val_accuracy.png"),
        title=f"Layer Probe [{depth}] — {model_name}",
    )
    plot_loss_curves(
        history["train_losses"], history["val_losses"],
        save_path=os.path.join(save_dir, "train_val_loss.png"),
        title=f"Layer Probe Loss [{depth}] — {model_name}",
    )

    # ── PCA embedding of the FIXED PROBE SUBSET (30 classes × 30 samples) ──
    fixed_loader = get_fixed_probe_subset(data_root, model_name, samples_per_class=30, seed=42)
    fixed_feats, fixed_labels = extract_layer_features(model, fixed_loader, model_name, depth, device)
    plot_embeddings(
        fixed_feats, fixed_labels, class_names,
        save_path=os.path.join(save_dir, "pca_embedding.png"),
        method="pca",
        title=f"PCA — {model_name} [{depth} layer]",
    )

    # ── Save metrics ─────────────────────────────────────────────────────────
    metrics = {
        "scenario": "layer_probe",
        "model": model_name,
        "depth": depth,
        "layer_name": LAYER_HOOKS[model_name][depth],
        "feature_dim": int(train_features.shape[1]),
        "best_val_acc": history["best_val_acc"],
        "feature_norms": feat_norms,
        "extraction_time": extraction_time,
        "total_train_time": history["total_train_time"],
        "avg_epoch_time": history["avg_epoch_time"],
        "total_scenario_time": time.time() - start_total,
    }
    save_metrics_json(metrics, os.path.join(save_dir, "metrics.json"))

    print(f"\n  ✓ Depth: {depth} → Layer: {LAYER_HOOKS[model_name][depth]}")
    print(f"  ✓ Best Val Accuracy: {history['best_val_acc']:.2f}%")
    print(f"  ✓ Results saved to: {save_dir}")
    return metrics
