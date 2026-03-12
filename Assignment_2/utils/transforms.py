"""
Per-model image transforms for training and validation.
- ResNet50 / DenseNet121: 224×224
- InceptionV3: 299×299
"""

from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

INPUT_SIZES = {
    "resnet50": 224,
    "inception_v3": 299,
    "densenet121": 224,
}


def get_transforms(model_name: str):
    """
    Returns (train_transform, val_transform) for the given model.
    """
    size = INPUT_SIZES.get(model_name, 224)

    train_transform = transforms.Compose([
        transforms.Resize((size + 32, size + 32)),
        transforms.RandomCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_transform, val_transform


def get_corruption_transform(model_name: str, corruption_type: str, severity=None):
    """
    Returns val transform with a corruption applied after tensor conversion.

    corruption_type: 'gaussian', 'motion_blur', 'brightness_dark', 'brightness_bright'
    severity: sigma value for gaussian (0.05, 0.10, 0.20) or None for others
    """
    import torch
    import torchvision.transforms.functional as TF
    from torchvision.transforms import GaussianBlur

    size = INPUT_SIZES.get(model_name, 224)
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    class AddGaussianNoise:
        def __init__(self, sigma):
            self.sigma = sigma
        def __call__(self, tensor):
            noise = torch.randn_like(tensor) * self.sigma
            return torch.clamp(tensor + noise, 0., 1.)

    class BrightnessShift:
        def __init__(self, factor):
            self.factor = factor
        def __call__(self, tensor):
            return torch.clamp(tensor * self.factor, 0., 1.)

    base = [
        transforms.Resize((size, size)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
    ]

    if corruption_type == "gaussian":
        sigma = severity if severity is not None else 0.1
        corrupt = [AddGaussianNoise(sigma)]
    elif corruption_type == "motion_blur":
        # Apply kernel blur at PIL stage before ToTensor
        corrupt = []
        base = [
            transforms.Resize((size, size)),
            transforms.CenterCrop(size),
            GaussianBlur(kernel_size=15, sigma=(3.0, 3.0)),  # simulates motion blur
            transforms.ToTensor(),
        ]
    elif corruption_type == "brightness_dark":
        factor = 0.5
        corrupt = [BrightnessShift(factor)]
    elif corruption_type == "brightness_bright":
        factor = 1.5
        corrupt = [BrightnessShift(factor)]
    else:
        corrupt = []

    return transforms.Compose(base + corrupt + [normalize])
