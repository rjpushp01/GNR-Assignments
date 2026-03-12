"""
AID Dataset loader with stratified train/val splits and subset sampling.
Supports 100%, 20%, and 5% data regimes for few-shot experiments.
"""

import os
import random
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from utils.transforms import get_transforms


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def stratified_split(dataset: ImageFolder, val_ratio: float = 0.3, seed: int = 42):
    """
    Stratified split of ImageFolder dataset into train and val indices.
    Returns (train_indices, val_indices).
    """
    set_seed(seed)
    label_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        label_to_indices[label].append(idx)

    train_indices, val_indices = [], []
    for label, indices in label_to_indices.items():
        indices = list(indices)
        random.shuffle(indices)
        n_val = max(1, int(len(indices) * val_ratio))
        val_indices.extend(indices[:n_val])
        train_indices.extend(indices[n_val:])

    return train_indices, val_indices


def subsample_indices(indices: list, dataset: ImageFolder, pct: float, seed: int = 42):
    """
    Subsample `pct` fraction of indices while keeping class balance (stratified).
    """
    if pct >= 1.0:
        return indices

    set_seed(seed)
    label_to_indices = defaultdict(list)
    for idx in indices:
        label = dataset.targets[idx]
        label_to_indices[label].append(idx)

    sampled = []
    for label, idxs in label_to_indices.items():
        idxs = list(idxs)
        random.shuffle(idxs)
        n = max(1, int(len(idxs) * pct))
        sampled.extend(idxs[:n])

    return sampled


def get_dataloaders(
    model_name: str,
    data_root: str,
    pct: float = 1.0,
    seed: int = 42,
    val_ratio: float = 0.3,
    batch_size: int = None,
    num_workers: int = 4,
):
    """
    Returns (train_loader, val_loader, class_names).

    Args:
        model_name: 'resnet50', 'inception_v3', or 'densenet121'
        data_root: Path to AID dataset root (parent of class folders)
        pct: Fraction of training data to use (1.0, 0.2, 0.05)
        seed: Random seed for reproducibility
        val_ratio: Fraction of full data to use as validation
        batch_size: Override batch size from config
        num_workers: DataLoader workers
    """
    from utils.model_utils import MODEL_CONFIGS

    train_transform, val_transform = get_transforms(model_name)

    # Load with train transforms first to get samples
    full_dataset_train = ImageFolder(root=data_root, transform=train_transform)
    full_dataset_val = ImageFolder(root=data_root, transform=val_transform)
    class_names = full_dataset_train.classes

    # Stratified split
    train_indices, val_indices = stratified_split(full_dataset_train, val_ratio, seed)

    # Subsample training data for few-shot
    if pct < 1.0:
        train_indices = subsample_indices(train_indices, full_dataset_train, pct, seed)

    train_dataset = Subset(full_dataset_train, train_indices)
    val_dataset = Subset(full_dataset_val, val_indices)

    cfg = MODEL_CONFIGS[model_name]
    bs = batch_size or cfg["batch_size"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, class_names


def get_fixed_probe_subset(data_root: str, model_name: str, samples_per_class: int = 30, seed: int = 42):
    """
    Returns a DataLoader with exactly `samples_per_class` per class.
    Same fixed subset across all models/layers for fair PCA comparison (Scenario 5).
    """
    from utils.transforms import get_transforms
    _, val_transform = get_transforms(model_name)
    dataset = ImageFolder(root=data_root, transform=val_transform)

    set_seed(seed)
    label_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        label_to_indices[label].append(idx)

    selected = []
    for label, indices in sorted(label_to_indices.items()):
        indices = list(indices)
        random.shuffle(indices)
        selected.extend(indices[:samples_per_class])

    return DataLoader(
        Subset(dataset, selected),
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
