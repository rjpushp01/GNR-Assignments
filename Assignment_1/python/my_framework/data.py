import os
import glob
import random
import my_backend as mb
try:
    from .tensor import Tensor
except ImportError:
    from tensor import Tensor

class DataLoader:
    def __init__(self, root_dir, batch_size=32, shuffle=True, flatten=False, mode='train', seed=42, augment=True):
        self.files = []
        self.labels = []
        
        if not os.path.exists(root_dir):
            raise ValueError(f"Directory {root_dir} does not exist")
            
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.flatten = flatten
        self.mode = mode
        self.augment = augment and (mode == 'train')  # Only augment during training
        
        print(f"Loading data from {root_dir}, found {len(self.classes)} classes. Mode: {mode}")
        
        all_files = []
        all_labels = []

        for cls in self.classes:
            cls_path = os.path.join(root_dir, cls)
            # Portable glob
            img_paths = sorted(glob.glob(os.path.join(cls_path, "*.png")) + glob.glob(os.path.join(cls_path, "*.jpg")))
            
            # Deterministic shuffle
            # Use random with seed
            rng = random.Random(seed)
            rng.shuffle(img_paths)
            
            # Split 70:20:10
            n = len(img_paths)
            n_train = int(0.7 * n)
            n_val = int(0.2 * n)
            
            if mode == 'train':
                selected = img_paths[:n_train]
            elif mode == 'val':
                selected = img_paths[n_train:n_train+n_val]
            elif mode == 'test':
                selected = img_paths[n_train+n_val:]
            else:
                selected = img_paths
            
            all_files.extend(selected)
            all_labels.extend([self.class_to_idx[cls]] * len(selected))
        
        self.files = all_files
        self.labels = all_labels
        self.num_samples = len(self.files)
        print(f"Total images in {mode} set: {self.num_samples}")
        if self.augment:
            print(f"  Data augmentation: ENABLED (random horizontal flip)")
        
    def __iter__(self):
        # Create indices
        indices = list(range(self.num_samples))
        if self.shuffle:
            random.shuffle(indices)
            
        for start_idx in range(0, self.num_samples, self.batch_size):
            batch_indices = indices[start_idx : start_idx + self.batch_size]
            
            batch_files = [self.files[i] for i in batch_indices]
            batch_labels = [self.labels[i] for i in batch_indices]
            
            is_mnist = 'data_1' in self.files[0]
            if is_mnist:
                H, W = 32, 32
                C = 1
            else:
                # CIFAR
                H, W = 32, 32
                C = 3
            
            # C++ Load Batch
            # Returns Tensor [B, C, H, W]
            raw_tensor = mb.ops.load_image_batch(batch_files, C, H, W)
            
            # Apply augmentation during training
            if self.augment:
                # Random crop with 4px padding (80% probability per image)
                # Pads to 40x40, then randomly crops back to 32x32
                raw_tensor = mb.ops.random_crop_with_padding(raw_tensor, 4, 0.8)
                # Random horizontal flip (50% chance per image)
                raw_tensor = mb.ops.random_horizontal_flip(raw_tensor, 0.5)
            
            images_tensor = Tensor(raw_tensor)
            
            # Labels Tensor
            labels_tensor = Tensor(batch_labels) # 1D list -> Tensor
            
            if self.flatten:
                pass

            yield images_tensor, labels_tensor

    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size
