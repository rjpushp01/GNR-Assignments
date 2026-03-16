"""
Shared training engine used by all scenarios.
Handles one epoch of training and one pass of validation.
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from utils.metrics import compute_accuracy, update_confusion_matrix, compute_gradient_norms, aggregate_grad_norms


def train_one_epoch(model, loader, criterion, optimizer, device, collect_grad_norms=False):
    model.train()
    total_loss, total_acc = 0.0, 0.0
    grad_norm_history = [] if collect_grad_norms else None
    
    data_fetch_time = 0.0
    t_start = time.time()
    t_iter = t_start

    for inputs, labels in tqdm(loader, desc="  Train", leave=False):
        data_fetch_time += time.time() - t_iter
        
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)

        # InceptionV3 returns (logits, aux_logits) during training
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        loss = criterion(outputs, labels)
        loss.backward()

        if collect_grad_norms:
            grad_norm_history.append(compute_gradient_norms(model))

        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        total_acc += compute_accuracy(outputs.detach(), labels) * inputs.size(0)
        t_iter = time.time()

    n = len(loader.dataset)
    return total_loss / n, total_acc / n, grad_norm_history, data_fetch_time


@torch.no_grad()
def validate(model, loader, criterion, device, num_classes=30, return_cm=False):
    model.eval()
    total_loss, total_acc = 0.0, 0.0
    cm = np.zeros((num_classes, num_classes), dtype=int) if return_cm else None
    all_features = []
    all_labels = []

    data_fetch_time = 0.0
    t_iter = time.time()
    for inputs, labels in tqdm(loader, desc="  Val  ", leave=False):
        data_fetch_time += time.time() - t_iter
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        outputs = model(inputs)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        loss = criterion(outputs, labels)
        total_loss += loss.item() * inputs.size(0)
        total_acc += compute_accuracy(outputs, labels) * inputs.size(0)
        if return_cm:
            update_confusion_matrix(cm, outputs, labels)
        t_iter = time.time()

    n = len(loader.dataset)
    return total_loss / n, total_acc / n, cm, data_fetch_time


def run_training(
    model,
    train_loader,
    val_loader,
    device,
    num_epochs: int,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    save_dir: str = ".",
    collect_grad_norms: bool = False,
    num_classes: int = 30,
    scheduler_type: str = "cosine",
):
    """
    Full training loop. Returns history dict.
    Saves best model checkpoint to save_dir/best_model.pth.
    """
    os.makedirs(save_dir, exist_ok=True)
    criterion = nn.CrossEntropyLoss()

    # Only optimize params with requires_grad=True
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)

    if scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    else:
        scheduler = None

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "epoch_times": [],
        "train_pass_times": [],
        "val_pass_times": [],
        "data_fetch_times": [],
    }
    all_grad_norms = []
    best_val_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        t_train_start = time.time()
        train_loss, train_acc, grad_norms, d_time_train = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            collect_grad_norms=collect_grad_norms,
        )
        t_train_end = time.time()
        
        t_val_start = time.time()
        val_loss, val_acc, _, d_time_val = validate(model, val_loader, criterion, device, num_classes)
        t_val_end = time.time()

        if scheduler:
            scheduler.step()

        epoch_time = time.time() - t_train_start
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["epoch_times"].append(epoch_time)
        history["train_pass_times"].append(t_train_end - t_train_start)
        history["val_pass_times"].append(t_val_end - t_val_start)
        history["data_fetch_times"].append(d_time_train + d_time_val)

        if collect_grad_norms and grad_norms:
            all_grad_norms.extend(grad_norms)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))

        print(f"  Epoch {epoch:3d}/{num_epochs} | "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.2f}%  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.2f}%  "
              f"[{epoch_time:.1f}s]")

    history["best_val_acc"] = best_val_acc
    history["total_time"] = sum(history["epoch_times"])
    history["total_train_pass_time"] = sum(history["train_pass_times"])
    history["total_val_pass_time"] = sum(history["val_pass_times"])
    history["total_data_fetch_time"] = sum(history["data_fetch_times"])
    history["avg_epoch_time"] = np.mean(history["epoch_times"])

    if collect_grad_norms and all_grad_norms:
        history["mean_grad_norms"] = aggregate_grad_norms(all_grad_norms)

    # Save history
    with open(os.path.join(save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    return history
