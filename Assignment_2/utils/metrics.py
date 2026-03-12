"""
Metrics computation: accuracy, confusion matrix, gradient norms.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple


def compute_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """Top-1 accuracy from logits."""
    preds = outputs.argmax(dim=1)
    return (preds == labels).float().mean().item() * 100.0


def update_confusion_matrix(
    cm: np.ndarray,
    outputs: torch.Tensor,
    labels: torch.Tensor,
) -> np.ndarray:
    """Increment confusion matrix in-place."""
    preds = outputs.argmax(dim=1).cpu().numpy()
    labels = labels.cpu().numpy()
    for p, t in zip(preds, labels):
        cm[t][p] += 1
    return cm


def compute_gradient_norms(model: nn.Module) -> Dict[str, float]:
    """
    Compute L2 gradient norm for each named parameter that has a gradient.
    Returns dict: {param_name: grad_norm}.
    """
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.data.norm(2).item()
    return grad_norms


def compute_feature_norms(features: np.ndarray) -> Dict[str, float]:
    """
    Compute per-sample and mean feature norms.
    features: (N, D) numpy array
    """
    norms = np.linalg.norm(features, axis=1)
    return {
        "mean_norm": float(norms.mean()),
        "std_norm": float(norms.std()),
        "min_norm": float(norms.min()),
        "max_norm": float(norms.max()),
    }


def aggregate_grad_norms(grad_norm_history: List[Dict]) -> Dict[str, float]:
    """
    Aggregate gradient norm history (list of per-step dicts) into mean per param.
    """
    aggregated = {}
    for step_norms in grad_norm_history:
        for name, val in step_norms.items():
            if name not in aggregated:
                aggregated[name] = []
            aggregated[name].append(val)
    return {name: float(np.mean(vals)) for name, vals in aggregated.items()}


def relative_performance_drop(acc_100: float, acc_5: float) -> float:
    """∆ = (Acc_100% - Acc_5%) / Acc_100%"""
    if acc_100 == 0:
        return 0.0
    return (acc_100 - acc_5) / acc_100


def corruption_error(acc_corrupted: float) -> float:
    """CE = 1 - Acc_corrupted (as fraction, not percentage)"""
    return 1.0 - (acc_corrupted / 100.0)


def relative_robustness(acc_corrupted: float, acc_clean: float) -> float:
    """Relative Robustness = Acc_corrupted / Acc_clean"""
    if acc_clean == 0:
        return 0.0
    return acc_corrupted / acc_clean
