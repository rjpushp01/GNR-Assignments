"""
Model utilities: loading pretrained models, freezing/unfreezing strategies,
parameter counting, MACs and FLOPs computation.
"""

import json
import os
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn

# Model configuration defaults
MODEL_CONFIGS = {
    "resnet50": {
        "timm_name": "resnet50",
        "input_size": 224,
        "batch_size": 64,
        "lr": 1e-3,
        "epochs_full": 30,
        "epochs_fewshot": 20,
    },
    "inception_v3": {
        "timm_name": "inception_v3",
        "input_size": 299,
        "batch_size": 32,
        "lr": 1e-3,
        "epochs_full": 30,
        "epochs_fewshot": 20,
    },
    "densenet121": {
        "timm_name": "densenet121",
        "input_size": 224,
        "batch_size": 64,
        "lr": 1e-3,
        "epochs_full": 30,
        "epochs_fewshot": 20,
    },
}


def load_model(model_name: str, num_classes: int = 30, pretrained: bool = True) -> nn.Module:
    """
    Load a pretrained model using timm and replace the classifier head.
    """
    import timm
    cfg = MODEL_CONFIGS[model_name]
    model = timm.create_model(
        cfg["timm_name"],
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return model


def freeze_backbone(model: nn.Module) -> nn.Module:
    """Freeze ALL parameters, then unfreeze only the classifier head."""
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the final classification layer
    # Works for resnet50, densenet121, inception_v3 via timm's .get_classifier()
    classifier = model.get_classifier()
    for param in classifier.parameters():
        param.requires_grad = True

    return model


def unfreeze_last_block(model: nn.Module, model_name: str) -> nn.Module:
    """Freeze all except the last convolutional block + classifier."""
    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze classifier
    classifier = model.get_classifier()
    for param in classifier.parameters():
        param.requires_grad = True

    # Unfreeze last block (model-specific layer names)
    last_blocks = {
        "resnet50": ["layer4"],
        "inception_v3": ["Mixed_7c", "Mixed_7b", "Mixed_7a"],
        "densenet121": ["features.denseblock4", "features.norm5"],
    }

    blocks = last_blocks.get(model_name, [])
    for name, module in model.named_modules():
        for block in blocks:
            if name.startswith(block):
                for param in module.parameters():
                    param.requires_grad = True

    return model


def unfreeze_full(model: nn.Module) -> nn.Module:
    """Unfreeze ALL parameters for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True
    return model


def selective_unfreeze(model: nn.Module, target_pct: float = 0.20) -> Dict:
    """
    Unfreeze the deepest layers until target_pct of total backbone params are trainable.
    Strategy: unfreeze from deepest to shallowest until budget exhausted.
    Returns dict with info about which layers were unfrozen.
    """
    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze classifier always
    classifier = model.get_classifier()
    for param in classifier.parameters():
        param.requires_grad = True

    # Count total backbone params (excluding classifier)
    classifier_params = set(id(p) for p in classifier.parameters())
    total_backbone = sum(
        p.numel() for p in model.parameters()
        if id(p) not in classifier_params
    )
    budget = int(total_backbone * target_pct)

    # Get named parameter groups (reversed = deepest first)
    named_params = [(name, param) for name, param in model.named_parameters()
                    if id(param) not in classifier_params]
    named_params = list(reversed(named_params))

    unfrozen_count = 0
    unfrozen_layers = []

    for name, param in named_params:
        if unfrozen_count + param.numel() <= budget:
            param.requires_grad = True
            unfrozen_count += param.numel()
            # Track layer names (parent module)
            layer_name = ".".join(name.split(".")[:-1])
            if layer_name and layer_name not in unfrozen_layers:
                unfrozen_layers.append(layer_name)
        else:
            break

    info = {
        "total_backbone_params": total_backbone,
        "unfrozen_params": unfrozen_count,
        "actual_pct": unfrozen_count / total_backbone * 100,
        "unfrozen_layers": unfrozen_layers[:20],  # top 20 for display
    }
    return info


def count_params(model: nn.Module) -> Dict:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "frozen_params": total - trainable,
        "trainable_pct": trainable / total * 100,
    }


def compute_macs_flops(model: nn.Module, model_name: str) -> Dict:
    """
    Compute MACs and FLOPs using ptflops.
    Returns dict with macs and flops (as strings and raw numbers).
    """
    try:
        from ptflops import get_model_complexity_info
        size = MODEL_CONFIGS[model_name]["input_size"]
        macs, params = get_model_complexity_info(
            model,
            (3, size, size),
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False,
        )
        return {
            "macs": macs,
            "flops": macs * 2,  # FLOPs ≈ 2 × MACs
            "macs_str": f"{macs/1e9:.3f} GMac",
            "flops_str": f"{macs*2/1e9:.3f} GFlops",
        }
    except Exception as e:
        return {"error": str(e), "macs": None, "flops": None}


def get_model_info(model: nn.Module, model_name: str, save_dir: str) -> Dict:
    """Compute and save full model efficiency info to JSON."""
    param_info = count_params(model)
    macs_info = compute_macs_flops(model, model_name)
    info = {
        "model_name": model_name,
        **param_info,
        **macs_info,
    }
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "model_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"  Total params:    {param_info['total_params']:,}")
    print(f"  Trainable params:{param_info['trainable_params']:,} ({param_info['trainable_pct']:.1f}%)")
    print(f"  MACs:            {macs_info.get('macs_str', 'N/A')}")
    print(f"  FLOPs:           {macs_info.get('flops_str', 'N/A')}")
    print(f"{'='*50}\n")
    return info


# Layer selection for Scenario 5 (Layer-Wise Feature Probing)
# Using block-level module names (reliable across timm versions).
# Rationale:
#   ResNet50   : layer1=early (64ch→256ch), layer2=mid (512ch), layer3=final (1024ch)
#   InceptionV3: Mixed_5c=early, Mixed_6e=mid, Mixed_7c=final
#   DenseNet121: denseblock1=early, denseblock2=mid, denseblock4=final
LAYER_HOOKS = {
    "resnet50": {
        "early": "layer1",
        "mid":   "layer2",
        "final": "layer3",
    },
    "inception_v3": {
        "early": "Mixed_5c",
        "mid":   "Mixed_6e",
        "final": "Mixed_7c",
    },
    "densenet121": {
        "early": "features.denseblock1",
        "mid":   "features.denseblock2",
        "final": "features.denseblock4",
    },
}


def register_hook(model: nn.Module, model_name: str, depth: str):
    """
    Register a forward hook on the specified layer depth.
    Returns (hook_handle, features_list).
    Tries exact match first, then prefix match as fallback.
    """
    layer_name = LAYER_HOOKS[model_name][depth]
    features_list = []

    def hook_fn(module, input, output):
        # Global average pool to get a flat feature vector
        if output.dim() == 4:
            feat = output.mean(dim=[2, 3])  # (B, C)
        elif output.dim() == 3:
            feat = output.mean(dim=1)       # (B, seq, C) -> (B, C)
        else:
            feat = output
        features_list.append(feat.detach().cpu())

    # Build lookup: exact match first, then startswith match
    all_named = list(model.named_modules())
    all_names = [n for n, _ in all_named]

    target_module = None
    matched_name = None

    # 1. Exact match
    for name, module in all_named:
        if name == layer_name:
            target_module = module
            matched_name = name
            break

    # 2. Prefix match (e.g. 'layer1' matches 'layer1' even if stored differently)
    if target_module is None:
        for name, module in all_named:
            if name.startswith(layer_name) and name != "":
                target_module = module
                matched_name = name
                break

    if target_module is None:
        top_names = [n for n in all_names if n][:30]
        raise ValueError(
            f"Layer '{layer_name}' not found in {model_name}.\n"
            f"Available top-level modules: {top_names}"
        )

    print(f"  Hook registered on: '{matched_name}'")
    handle = target_module.register_forward_hook(hook_fn)
    return handle, features_list
