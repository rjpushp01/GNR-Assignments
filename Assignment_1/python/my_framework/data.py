import os
import cv2
import numpy as np
import glob
try:
    from .tensor import Tensor
except ImportError:
    from tensor import Tensor

class DataLoader:
    def __init__(self, root_dir, batch_size=32, shuffle=True, flatten=False, mode='train', seed=42):
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
        
        print(f"Loading data from {root_dir}, found {len(self.classes)} classes. Mode: {mode}")
        
        all_files = []
        all_labels = []

        for cls in self.classes:
            cls_path = os.path.join(root_dir, cls)
            img_paths = sorted(glob.glob(os.path.join(cls_path, "*.png")) + glob.glob(os.path.join(cls_path, "*.jpg")))
            
            # Deterministic shuffle for splitting
            rng = np.random.RandomState(seed)
            rng.shuffle(img_paths)
            
            # Split 70:20:10
            n = len(img_paths)
            n_train = int(0.7 * n)
            n_val = int(0.2 * n)
            # n_test = rest
            
            if mode == 'train':
                selected = img_paths[:n_train]
            elif mode == 'val':
                selected = img_paths[n_train:n_train+n_val]
            elif mode == 'test':
                selected = img_paths[n_train+n_val:]
            else:
                 # Fallback or 'all'
                selected = img_paths
            
            all_files.extend(selected)
            all_labels.extend([self.class_to_idx[cls]] * len(selected))
        
        self.files = np.array(all_files)
        self.labels = np.array(all_labels)
        self.num_samples = len(self.files)
        print(f"Total images in {mode} set: {self.num_samples}")
        
    def __iter__(self):
        indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(indices)
            
        for start_idx in range(0, self.num_samples, self.batch_size):
            batch_idx = indices[start_idx : start_idx + self.batch_size]
            batch_files = self.files[batch_idx]
            batch_labels = self.labels[batch_idx]
            
            imgs = []
            for f in batch_files:
                # Read logic
                if 'data_1' in f:
                    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                else:
                    img = cv2.imread(f, cv2.IMREAD_COLOR)

                if img is None: continue
                
                # Check shape correctness
                # Data 1: 28x28 (Gray)
                # Data 2: 32x32? (RGB)
                
                img = img.astype(np.float32) / 255.0
                
                if self.flatten:
                    img = img.flatten()
                else:
                    # HWC -> CHW
                    if len(img.shape) == 3:
                        img = np.transpose(img, (2, 0, 1)) # C,H,W
                    else:
                         img = img[np.newaxis, :, :] # 1,H,W for grayscale
                
                imgs.append(img)
            
            if not imgs: continue
            
            yield Tensor(np.array(imgs)), Tensor(np.array(batch_labels))

    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size
