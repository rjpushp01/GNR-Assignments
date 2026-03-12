"""
Visualization utilities: accuracy curves, confusion matrices, embeddings (PCA/t-SNE/UMAP),
gradient norm plots, and layer-wise feature visualizations.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from typing import Dict, List, Optional


# ── Styling ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]


# ── Accuracy curves ───────────────────────────────────────────────────────────
def plot_accuracy_curves(
    train_accs: List[float],
    val_accs: List[float],
    save_path: str,
    title: str = "Training & Validation Accuracy",
):
    epochs = list(range(1, len(train_accs) + 1))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_accs, label="Train", color=PALETTE[0], linewidth=2, marker="o", markersize=4)
    ax.plot(epochs, val_accs, label="Validation", color=PALETTE[1], linewidth=2, marker="s", markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def plot_loss_curves(
    train_losses: List[float],
    val_losses: List[float],
    save_path: str,
    title: str = "Training & Validation Loss",
):
    epochs = list(range(1, len(train_losses) + 1))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, label="Train Loss", color=PALETTE[0], linewidth=2, marker="o", markersize=4)
    ax.plot(epochs, val_losses, label="Val Loss", color=PALETTE[1], linewidth=2, marker="s", markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


# ── Confusion matrix ──────────────────────────────────────────────────────────
def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str,
    title: str = "Confusion Matrix",
    normalize: bool = True,
):
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_plot = cm.astype(float) / (row_sums + 1e-8)
    else:
        cm_plot = cm.astype(float)

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm_plot, annot=False, fmt=".2f" if normalize else "d",
        cmap="Blues", ax=ax,
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.3
    )
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", rotation=0, labelsize=7)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


# ── Feature embeddings (PCA / t-SNE / UMAP) ──────────────────────────────────
def plot_embeddings(
    features: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    save_path: str,
    method: str = "tsne",
    title: str = None,
):
    """
    Project features to 2D using PCA, t-SNE, or UMAP and scatter plot by class.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    # Normalise
    features = StandardScaler().fit_transform(features)

    # First reduce to 50 dims with PCA for speed
    n_components_pca = min(50, features.shape[0] - 1, features.shape[1])
    pca50 = PCA(n_components=n_components_pca, random_state=42)
    reduced = pca50.fit_transform(features)

    if method == "pca":
        coords = reduced[:, :2]
        label_method = "PCA"
    elif method == "tsne":
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        coords = tsne.fit_transform(reduced)
        label_method = "t-SNE"
    elif method == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42)
            coords = reducer.fit_transform(reduced)
        except ImportError:
            from sklearn.manifold import TSNE
            coords = TSNE(n_components=2, random_state=42).fit_transform(reduced)
        label_method = "UMAP"
    else:
        raise ValueError(f"Unknown method: {method}")

    n_classes = len(class_names)
    cmap = plt.cm.get_cmap("tab20", n_classes)

    fig, ax = plt.subplots(figsize=(10, 8))
    for cls_idx in range(n_classes):
        mask = labels == cls_idx
        if mask.sum() == 0:
            continue
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            s=20, alpha=0.7, color=cmap(cls_idx),
            label=class_names[cls_idx] if cls_idx < 15 else None,
        )
    ax.set_title(title or f"{label_method} Feature Embedding")
    ax.set_xlabel(f"{label_method}-1")
    ax.set_ylabel(f"{label_method}-2")
    if n_classes <= 15:
        ax.legend(fontsize=7, ncol=2, loc="best", markerscale=1.5)
    else:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, n_classes))
        plt.colorbar(sm, ax=ax, label="Class index")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


# ── Gradient norms ────────────────────────────────────────────────────────────
def plot_gradient_norms(
    grad_norms: Dict[str, float],
    save_path: str,
    title: str = "Mean Gradient Norms by Layer",
    top_n: int = 40,
):
    """Bar chart of gradient norms per layer (top_n layers)."""
    items = sorted(grad_norms.items(), key=lambda x: x[1], reverse=True)[:top_n]
    if not items:
        return
    names = [it[0].replace(".", "\n") for it in items]
    values = [it[1] for it in items]

    fig, ax = plt.subplots(figsize=(max(12, top_n * 0.4), 6))
    bars = ax.bar(range(len(values)), values, color=PALETTE[0], alpha=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=6)
    ax.set_ylabel("Gradient L2 Norm")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.4)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


# ── Fine-tuning strategy comparison ──────────────────────────────────────────
def plot_finetune_comparison(
    strategy_results: Dict[str, Dict],
    save_path: str,
    title: str = "Accuracy vs. % Unfrozen Parameters",
):
    """
    strategy_results: {strategy_name: {"trainable_pct": x, "best_val_acc": y}}
    """
    strategies = list(strategy_results.keys())
    pcts = [strategy_results[s]["trainable_pct"] for s in strategies]
    accs = [strategy_results[s]["best_val_acc"] for s in strategies]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(pcts, accs, marker="o", linewidth=2, color=PALETTE[0], markersize=10)
    for i, (s, p, a) in enumerate(zip(strategies, pcts, accs)):
        ax.annotate(s, (p, a), textcoords="offset points", xytext=(5, 5), fontsize=9)
    ax.set_xlabel("% Trainable Parameters")
    ax.set_ylabel("Best Validation Accuracy (%)")
    ax.set_title(title)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


# ── Few-shot bar chart ────────────────────────────────────────────────────────
def plot_fewshot_results(
    results: Dict,
    save_path: str,
    model_name: str = "",
):
    """
    results: {"pct_5": acc, "pct_20": acc, "pct_100": acc}
    """
    labels = ["5%", "20%", "100%"]
    keys = ["pct_5", "pct_20", "pct_100"]
    values = [results.get(k, 0) for k in keys]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, values, color=[PALETTE[2], PALETTE[1], PALETTE[0]], alpha=0.85, width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_xlabel("Training Data %")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title(f"Few-Shot Performance — {model_name}")
    ax.set_ylim(0, min(100, max(values) + 10))
    ax.grid(True, axis="y", alpha=0.4)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


# ── Corruption robustness heatmap ─────────────────────────────────────────────
def plot_corruption_heatmap(
    corruption_data: Dict[str, float],
    save_path: str,
    model_name: str = "",
):
    """
    corruption_data: {"gaussian_0.05": acc, "gaussian_0.10": acc, ...}
    """
    names = list(corruption_data.keys())
    accs = [corruption_data[k] for k in names]
    ces = [1 - a / 100 for a in accs]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy bar
    colors = [PALETTE[0] if a > 60 else PALETTE[3] for a in accs]
    axes[0].barh(names, accs, color=colors, alpha=0.85)
    axes[0].set_xlabel("Validation Accuracy (%)")
    axes[0].set_title(f"Accuracy Under Corruptions — {model_name}")
    axes[0].axvline(x=0, color="black", linewidth=0.5)
    for i, v in enumerate(accs):
        axes[0].text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9)
    axes[0].grid(True, axis="x", alpha=0.4)

    # Corruption error bar
    axes[1].barh(names, ces, color=PALETTE[3], alpha=0.7)
    axes[1].set_xlabel("Corruption Error (1 - Acc)")
    axes[1].set_title("Corruption Error")
    for i, v in enumerate(ces):
        axes[1].text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=9)
    axes[1].grid(True, axis="x", alpha=0.4)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


# ── Layer-wise accuracy vs depth ──────────────────────────────────────────────
def plot_accuracy_vs_depth(
    depth_results: Dict[str, float],
    save_path: str,
    model_name: str = "",
):
    """
    depth_results: {"early": acc, "mid": acc, "final": acc}
    """
    depths = list(depth_results.keys())
    accs = [depth_results[d] for d in depths]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(range(len(depths)), accs, marker="o", linewidth=2.5, color=PALETTE[0], markersize=10)
    ax.set_xticks(range(len(depths)))
    ax.set_xticklabels(depths)
    ax.set_xlabel("Layer Depth")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title(f"Linear Probe Accuracy vs Depth — {model_name}")
    for i, (d, a) in enumerate(zip(depths, accs)):
        ax.annotate(f"{a:.1f}%", (i, a), textcoords="offset points",
                    xytext=(5, 5), fontsize=10)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def save_metrics_json(metrics: Dict, save_path: str):
    """Helper to save metrics dict as JSON."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics → {save_path}")
