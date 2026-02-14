"""
Model analysis utilities: parameter counting, MACs/FLOPs computation, and model summary.
Part of the custom deep learning framework for GNR 638 Assignment 1.
"""

import sys

# ---- Layer-type imports ----
from .models import Module, Conv2d, Linear, ReLU, MaxPool2d, Flatten, GlobalAvgPool2d


def count_parameters(model):
    """Count total trainable parameters in a model."""
    total = 0
    for p in model.parameters():
        size = 1
        for s in p.shape:
            size *= s
        total += size
    return total


def _conv2d_output_shape(h_in, w_in, kernel_size, stride, padding):
    h_out = (h_in + 2 * padding - kernel_size) // stride + 1
    w_out = (w_in + 2 * padding - kernel_size) // stride + 1
    return h_out, w_out


def _maxpool_output_shape(h_in, w_in, kernel_size, stride):
    h_out = (h_in - kernel_size) // stride + 1
    w_out = (w_in - kernel_size) // stride + 1
    return h_out, w_out


def compute_layer_stats(model, input_shape):
    """
    Walk through a model's layers and compute per-layer statistics.

    Args:
        model: A Module instance (MNIST_Model or CIFAR_Model).
        input_shape: Tuple (C, H, W) — single sample without batch dim.

    Returns:
        List of dicts with keys:
            name, type, output_shape, params, macs, flops
        And a summary dict with totals.
    """
    layers = []
    # Collect ordered layers from model attributes
    for attr_name in model.__dict__:
        layer = getattr(model, attr_name)
        if isinstance(layer, Module):
            layers.append((attr_name, layer))

    stats = []
    c, h, w = input_shape  # current feature map shape

    total_params = 0
    total_macs = 0
    total_flops = 0

    for name, layer in layers:
        params = 0
        macs = 0

        if isinstance(layer, Conv2d):
            # Weight shape: (out_ch, in_ch, kH, kW)
            out_ch = layer.weight.shape[0]
            in_ch = layer.weight.shape[1]
            kH = layer.weight.shape[2]
            kW = layer.weight.shape[3]
            stride = layer.stride
            padding = layer.padding

            h_out, w_out = _conv2d_output_shape(h, w, kH, stride, padding)

            # Params = out_ch * in_ch * kH * kW  (no bias in current Conv2d)
            params = out_ch * in_ch * kH * kW
            # MACs = out_ch * h_out * w_out * (in_ch * kH * kW)
            macs = out_ch * h_out * w_out * (in_ch * kH * kW)

            c, h, w = out_ch, h_out, w_out
            layer_type = "Conv2d"
            out_shape = f"[B, {c}, {h}, {w}]"

        elif isinstance(layer, Linear):
            in_f = layer.weight.shape[0]
            out_f = layer.weight.shape[1]
            # Params = in_f * out_f + out_f (bias)
            params = in_f * out_f + out_f
            # MACs = in_f * out_f
            macs = in_f * out_f

            c = out_f
            layer_type = "Linear"
            out_shape = f"[B, {out_f}]"

        elif isinstance(layer, ReLU):
            # Element-wise comparison — no MACs conventionally
            layer_type = "ReLU"
            if w > 0:
                out_shape = f"[B, {c}, {h}, {w}]"
            else:
                out_shape = f"[B, {c}]"

        elif isinstance(layer, MaxPool2d):
            ksize = layer.kernel_size
            stride = layer.stride
            h_out, w_out = _maxpool_output_shape(h, w, ksize, stride)
            h, w = h_out, w_out
            layer_type = "MaxPool2d"
            out_shape = f"[B, {c}, {h}, {w}]"

        elif isinstance(layer, Flatten):
            flat_size = c * h * w
            c = flat_size
            h = 0
            w = 0
            layer_type = "Flatten"
            out_shape = f"[B, {flat_size}]"

        elif isinstance(layer, GlobalAvgPool2d):
            # [B, C, H, W] -> [B, C]
            layer_type = "GlobalAvgPool2d"
            out_shape = f"[B, {c}]"
            h = 0
            w = 0

        else:
            layer_type = type(layer).__name__
            out_shape = "?"

        flops = 2 * macs  # 1 multiply + 1 add per MAC

        total_params += params
        total_macs += macs
        total_flops += flops

        stats.append({
            "name": name,
            "type": layer_type,
            "output_shape": out_shape,
            "params": params,
            "macs": macs,
            "flops": flops,
        })

    summary = {
        "total_params": total_params,
        "total_macs": total_macs,
        "total_flops": total_flops,
    }

    return stats, summary


def _fmt(n):
    """Format large numbers with commas."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def print_model_summary(model, input_shape, file=None):
    """
    Print a formatted model summary table showing per-layer params/MACs/FLOPs.

    Args:
        model: Module instance.
        input_shape: (C, H, W) for a single sample.
        file: Optional file path to also write the summary to.
    """
    stats, summary = compute_layer_stats(model, input_shape)

    header = f"{'Layer':<16} {'Type':<12} {'Output Shape':<18} {'Params':>10} {'MACs':>12} {'FLOPs':>12}"
    sep = "-" * len(header)

    lines = []
    lines.append("")
    lines.append("=" * len(header))
    lines.append(f"  MODEL SUMMARY — {type(model).__name__}")
    lines.append(f"  Input shape: [B, {input_shape[0]}, {input_shape[1]}, {input_shape[2]}]")
    lines.append("=" * len(header))
    lines.append(header)
    lines.append(sep)

    for s in stats:
        line = f"{s['name']:<16} {s['type']:<12} {s['output_shape']:<18} {s['params']:>10,} {s['macs']:>12,} {s['flops']:>12,}"
        lines.append(line)

    lines.append(sep)
    lines.append(
        f"{'TOTAL':<16} {'':<12} {'':<18} {summary['total_params']:>10,} {summary['total_macs']:>12,} {summary['total_flops']:>12,}"
    )
    lines.append(sep)
    lines.append(f"  Total trainable parameters : {summary['total_params']:,}")
    lines.append(f"  Total MACs (per forward)   : {_fmt(summary['total_macs'])}")
    lines.append(f"  Total FLOPs (per forward)  : {_fmt(summary['total_flops'])}")
    lines.append("=" * len(header))
    lines.append("")

    text = "\n".join(lines)
    print(text)

    if file:
        with open(file, "w") as f:
            f.write(text)
        print(f"  Model summary saved to {file}")

    return summary
