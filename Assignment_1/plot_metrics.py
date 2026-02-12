"""
Generate comprehensive training metric plots from the CSV logs produced by train.py.
Produces individual high-quality plots and a combined dashboard for documentation.

Usage:
    python plot_metrics.py --log_file results/training_log_mnist.csv
    python plot_metrics.py --log_file results/training_log_cifar.csv
"""
import csv
import argparse
import os
import math

# ── Minimal plotting with matplotlib only ──
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec


# ── Color Palette (Material-inspired) ──
COLORS = {
    'train_loss':  '#1565C0',   # Blue 800
    'val_loss':    '#C62828',   # Red 800
    'train_acc':   '#2E7D32',   # Green 800
    'val_acc':     '#E65100',   # Orange 900
    'lr':          '#6A1B9A',   # Purple 800
    'memory':      '#00838F',   # Cyan 800
    'epoch_time':  '#7B1FA2',   # Purple 700
    'batch_time':  '#AD1457',   # Pink 800
    'gap':         '#EF5350',   # Red 400
    'fill_loss':   '#BBDEFB',
    'fill_acc':    '#C8E6C9',
    'fill_mem':    '#B2EBF2',
    'fill_gap':    '#FFCDD2',
    'cum_time':    '#D84315',   # Deep Orange 800
}


def read_csv_log(path):
    """Read training log CSV and return list of dicts."""
    rows = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                k = k.strip()
                try:
                    parsed[k] = float(v)
                except (ValueError, TypeError):
                    parsed[k] = v
            rows.append(parsed)
    return rows


def _style_ax(ax, title, xlabel, ylabel, fontsize=12):
    ax.set_title(title, fontsize=fontsize + 2, fontweight='bold', pad=10)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=fontsize - 1)


def _get(rows, key):
    return [r.get(key, 0) for r in rows]


def plot_metrics(log_file):
    if not os.path.exists(log_file):
        print(f"Error: {log_file} not found. Run training first.")
        return

    rows = read_csv_log(log_file)
    if not rows:
        print("Error: log file is empty.")
        return

    if not os.path.exists("results"):
        os.makedirs("results")

    base_name = os.path.splitext(os.path.basename(log_file))[0]
    
    # Infer dataset name for titles
    if 'cifar' in base_name.lower():
        dataset_label = 'CIFAR-100'
    elif 'mnist' in base_name.lower():
        dataset_label = 'MNIST'
    else:
        dataset_label = base_name

    epochs = _get(rows, 'Epoch')
    train_loss = _get(rows, 'Train Loss')
    val_loss = _get(rows, 'Val Loss')
    train_acc = _get(rows, 'Train Acc')
    val_acc = _get(rows, 'Val Acc')
    epoch_time = _get(rows, 'Epoch Time (s)')
    cum_time = _get(rows, 'Cumulative Time (s)')
    avg_batch_time = _get(rows, 'Avg Batch Time (s)')
    lr = _get(rows, 'Learning Rate')
    memory = _get(rows, 'Memory (MB)')

    has_lr = any(v > 0 for v in lr)
    has_memory = any(v > 0 for v in memory)
    has_epoch_time = any(v > 0 for v in epoch_time)
    has_cum_time = any(v > 0 for v in cum_time)
    has_batch_time = any(v > 0 for v in avg_batch_time)
    num_epochs = len(epochs)

    saved_files = []

    # ================================================================
    #  1.  LOSS CURVE
    # ================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_loss, 'o-', label='Train Loss', color=COLORS['train_loss'], linewidth=2.5, markersize=6)
    ax.plot(epochs, val_loss, 's-', label='Val Loss', color=COLORS['val_loss'], linewidth=2.5, markersize=6)
    ax.fill_between(epochs, train_loss, val_loss, alpha=0.08, color=COLORS['val_loss'])
    _style_ax(ax, f'{dataset_label} — Training & Validation Loss', 'Epoch', 'Loss')
    ax.legend(fontsize=12, loc='best', framealpha=0.9)
    if num_epochs > 1:
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig.tight_layout()
    path = f"results/loss_curve_{base_name}.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    saved_files.append(path)
    print(f"  ✓ {path}")

    # ================================================================
    #  2.  ACCURACY CURVE
    # ================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, [a * 100 for a in train_acc], 'o-', label='Train Acc', color=COLORS['train_acc'], linewidth=2.5, markersize=6)
    ax.plot(epochs, [a * 100 for a in val_acc], 's-', label='Val Acc', color=COLORS['val_acc'], linewidth=2.5, markersize=6)
    ax.fill_between(epochs, [a * 100 for a in train_acc], [a * 100 for a in val_acc], alpha=0.08, color=COLORS['val_acc'])
    _style_ax(ax, f'{dataset_label} — Training & Validation Accuracy', 'Epoch', 'Accuracy (%)')
    ax.set_ylim(0, 105)
    ax.legend(fontsize=12, loc='best', framealpha=0.9)
    if num_epochs > 1:
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig.tight_layout()
    path = f"results/accuracy_curve_{base_name}.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    saved_files.append(path)
    print(f"  ✓ {path}")

    # ================================================================
    #  3.  OVERFITTING GAP (Train Acc - Val Acc)
    # ================================================================
    gap = [(ta - va) * 100 for ta, va in zip(train_acc, val_acc)]
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_colors = [COLORS['gap'] if g > 10 else '#66BB6A' for g in gap]
    ax.bar(epochs, gap, color=bar_colors, alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.axhline(y=10, color='#E53935', linestyle='--', linewidth=1.5, alpha=0.7, label='Overfitting threshold (10%)')
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    _style_ax(ax, f'{dataset_label} — Generalization Gap (Train Acc − Val Acc)', 'Epoch', 'Gap (%)')
    ax.legend(fontsize=11, loc='best')
    if num_epochs > 1:
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig.tight_layout()
    path = f"results/overfitting_gap_{base_name}.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    saved_files.append(path)
    print(f"  ✓ {path}")

    # ================================================================
    #  4.  LEARNING RATE SCHEDULE
    # ================================================================
    if has_lr:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, lr, 'D-', color=COLORS['lr'], linewidth=2.5, markersize=8, label='Learning Rate')
        ax.fill_between(epochs, lr, alpha=0.12, color=COLORS['lr'])
        _style_ax(ax, f'{dataset_label} — Learning Rate Schedule', 'Epoch', 'Learning Rate')
        ax.legend(fontsize=12)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(-3, -3))
        if num_epochs > 1:
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        fig.tight_layout()
        path = f"results/lr_schedule_{base_name}.png"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        saved_files.append(path)
        print(f"  ✓ {path}")

    # ================================================================
    #  5.  MEMORY USAGE
    # ================================================================
    if has_memory:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, memory, 'o-', color=COLORS['memory'], linewidth=2.5, markersize=6, label='Peak RSS')
        ax.fill_between(epochs, memory, alpha=0.12, color=COLORS['memory'])
        _style_ax(ax, f'{dataset_label} — Memory Usage (Peak RSS)', 'Epoch', 'Memory (MB)')
        ax.legend(fontsize=12)
        # Nice y limits
        if memory:
            ymin = min(memory) * 0.9
            ymax = max(memory) * 1.1
            ax.set_ylim(ymin, ymax)
        if num_epochs > 1:
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        fig.tight_layout()
        path = f"results/memory_usage_{base_name}.png"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        saved_files.append(path)
        print(f"  ✓ {path}")

    # ================================================================
    #  6.  EPOCH TIME + BATCH TIME
    # ================================================================
    if has_epoch_time:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        bars = ax1.bar(epochs, epoch_time, color=COLORS['epoch_time'], alpha=0.75, edgecolor='white', linewidth=0.5, label='Epoch Time')
        _style_ax(ax1, f'{dataset_label} — Training Time per Epoch', 'Epoch', 'Epoch Time (s)')

        if has_batch_time:
            ax2 = ax1.twinx()
            ax2.plot(epochs, [bt * 1000 for bt in avg_batch_time], 'D-', color=COLORS['batch_time'], linewidth=2, markersize=7, label='Avg Batch Time')
            ax2.set_ylabel('Avg Batch Time (ms)', fontsize=12, color=COLORS['batch_time'])
            ax2.tick_params(axis='y', labelcolor=COLORS['batch_time'])
            ax2.spines['top'].set_visible(False)
            # Combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=11, loc='upper right')
        else:
            ax1.legend(fontsize=11)

        if num_epochs > 1:
            ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        fig.tight_layout()
        path = f"results/epoch_time_{base_name}.png"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        saved_files.append(path)
        print(f"  ✓ {path}")

    # ================================================================
    #  7.  CUMULATIVE TIME
    # ================================================================
    if has_cum_time:
        fig, ax = plt.subplots(figsize=(10, 6))
        cum_min = [t / 60.0 for t in cum_time]
        ax.plot(epochs, cum_min, 'o-', color=COLORS['cum_time'], linewidth=2.5, markersize=6)
        ax.fill_between(epochs, cum_min, alpha=0.12, color=COLORS['cum_time'])
        _style_ax(ax, f'{dataset_label} — Cumulative Training Time', 'Epoch', 'Time (minutes)')
        if num_epochs > 1:
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        fig.tight_layout()
        path = f"results/cumulative_time_{base_name}.png"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        saved_files.append(path)
        print(f"  ✓ {path}")

    # ================================================================
    #  8.  COMBINED DASHBOARD (3×2 grid)
    # ================================================================
    fig = plt.figure(figsize=(18, 16))
    fig.suptitle(f'Training Dashboard — {dataset_label}', fontsize=20, fontweight='bold', y=0.98)
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.30)

    # (0,0) Loss
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(epochs, train_loss, 'o-', label='Train', color=COLORS['train_loss'], linewidth=2)
    ax.plot(epochs, val_loss, 's-', label='Val', color=COLORS['val_loss'], linewidth=2)
    ax.fill_between(epochs, train_loss, val_loss, alpha=0.08, color=COLORS['val_loss'])
    _style_ax(ax, 'Loss', 'Epoch', 'Loss', fontsize=11)
    ax.legend(fontsize=10)

    # (0,1) Accuracy
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(epochs, [a * 100 for a in train_acc], 'o-', label='Train', color=COLORS['train_acc'], linewidth=2)
    ax.plot(epochs, [a * 100 for a in val_acc], 's-', label='Val', color=COLORS['val_acc'], linewidth=2)
    ax.fill_between(epochs, [a * 100 for a in train_acc], [a * 100 for a in val_acc], alpha=0.08, color=COLORS['val_acc'])
    _style_ax(ax, 'Accuracy', 'Epoch', 'Accuracy (%)', fontsize=11)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10)

    # (1,0) Overfitting Gap
    ax = fig.add_subplot(gs[1, 0])
    bar_colors = [COLORS['gap'] if g > 10 else '#66BB6A' for g in gap]
    ax.bar(epochs, gap, color=bar_colors, alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.axhline(y=10, color='#E53935', linestyle='--', linewidth=1.2, alpha=0.7)
    _style_ax(ax, 'Generalization Gap', 'Epoch', 'Train−Val Acc Gap (%)', fontsize=11)

    # (1,1) Learning Rate
    ax = fig.add_subplot(gs[1, 1])
    if has_lr:
        ax.plot(epochs, lr, 'D-', color=COLORS['lr'], linewidth=2, markersize=6)
        ax.fill_between(epochs, lr, alpha=0.12, color=COLORS['lr'])
        _style_ax(ax, 'Learning Rate', 'Epoch', 'LR', fontsize=11)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(-3, -3))
    else:
        ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes, fontsize=16, color='gray')
        _style_ax(ax, 'Learning Rate', 'Epoch', 'LR', fontsize=11)

    # (2,0) Epoch Time
    ax = fig.add_subplot(gs[2, 0])
    if has_epoch_time:
        ax.bar(epochs, epoch_time, color=COLORS['epoch_time'], alpha=0.75, edgecolor='white', linewidth=0.5)
        if has_batch_time:
            ax2 = ax.twinx()
            ax2.plot(epochs, [bt * 1000 for bt in avg_batch_time], 'D-', color=COLORS['batch_time'], linewidth=1.5, markersize=5)
            ax2.set_ylabel('Batch (ms)', fontsize=10, color=COLORS['batch_time'])
            ax2.tick_params(axis='y', labelcolor=COLORS['batch_time'], labelsize=9)
    _style_ax(ax, 'Epoch Time', 'Epoch', 'Time (s)', fontsize=11)

    # (2,1) Memory Usage
    ax = fig.add_subplot(gs[2, 1])
    if has_memory:
        ax.plot(epochs, memory, 'o-', color=COLORS['memory'], linewidth=2, markersize=5)
        ax.fill_between(epochs, memory, alpha=0.12, color=COLORS['memory'])
        _style_ax(ax, 'Memory (Peak RSS)', 'Epoch', 'MB', fontsize=11)
        if memory:
            ymin = min(memory) * 0.9
            ymax = max(memory) * 1.1
            ax.set_ylim(ymin, ymax)
    elif has_cum_time:
        cum_min = [t / 60.0 for t in cum_time]
        ax.plot(epochs, cum_min, 'o-', color=COLORS['cum_time'], linewidth=2)
        ax.fill_between(epochs, cum_min, alpha=0.12, color=COLORS['cum_time'])
        _style_ax(ax, 'Cumulative Time', 'Epoch', 'Minutes', fontsize=11)
    else:
        ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes, fontsize=16, color='gray')
        _style_ax(ax, 'Memory', 'Epoch', 'MB', fontsize=11)

    path = f"results/combined_dashboard_{base_name}.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    saved_files.append(path)
    print(f"  ✓ {path}")

    # ================================================================
    #  SUMMARY
    # ================================================================
    print(f"\n{'═' * 50}")
    print(f"  {len(saved_files)} plots saved to 'results/' directory")
    print(f"{'═' * 50}")
    for f in saved_files:
        print(f"    • {f}")
    print()

    # Print key stats
    best_val_idx = val_acc.index(max(val_acc))
    final_gap = gap[-1] if gap else 0
    print(f"  Best Val Accuracy : {val_acc[best_val_idx]*100:.2f}% (Epoch {int(epochs[best_val_idx])})")
    print(f"  Final Train Acc   : {train_acc[-1]*100:.2f}%")
    print(f"  Final Val Acc     : {val_acc[-1]*100:.2f}%")
    print(f"  Final Overfit Gap : {final_gap:.1f}%")
    if has_cum_time:
        print(f"  Total Train Time  : {cum_time[-1]/60:.1f} min")
    if has_memory:
        print(f"  Peak Memory       : {max(memory):.0f} MB")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate training metric plots from CSV logs")
    parser.add_argument('--log_file', type=str, default='results/training_log_mnist.csv',
                        help="Path to the training log CSV file")
    args = parser.parse_args()
    plot_metrics(args.log_file)
