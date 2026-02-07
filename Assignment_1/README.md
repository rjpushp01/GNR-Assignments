# GNR 638 Assignment 1: Deep Learning Framework

This project implements a custom Deep Learning framework from scratch using C++ for the backend (math operations) and Python for the frontend (autograd, model definition), as per the assignment requirements.

## 1. Setup

### Environment
Ensure you have the conda environment set up:
```bash
conda env create -f environment.yml
conda activate gnr638_ass
```

### Building the C++ Backend
The core mathematics is implemented in C++ and needs to be compiled.
```bash
cmake -S . -B build
cmake --build build
```
This will compile the `my_backend` shared object and place it in `python/my_framework/`.

## 2. Project Structure
- `src/`: C++ Source code (`tensor.cpp`, `ops.cpp`, `bindings.cpp`).
- `include/`: C++ Headers.
- `python/my_framework/`: Python package.
  - `tensor.py`: Wrapper for C++ Tensor and Autograd engine.
  - `models.py`: Layer and Model definitions (Conv2d, Linear, MNIST_Model, etc.).
  - `optim.py`: SGD Optimizer.
  - `data.py`: DataLoader using OpenCV.
- `tests/`: Verification scripts.

## 3. Running Training

### Dataset 1 (MNIST)
To train the model on the first dataset (`data_1`):
```bash
python train.py --dataset mnist --data_path data_1 --epochs 5 --lr 0.01
```

### Dataset 2 (CIFAR-100)
To train the model on the second dataset (`data_2`):
```bash
python train.py --dataset cifar --data_path data_2 --epochs 10 --lr 0.005
```

## 4. Features
- **Hybrid Architecture**: C++ backend for performance, Python for flexibility.
- **Autograd**: Custom DAG-based automatic differentiation.
- **Optimized Ops**: Im2Col implementation for Convolution, tiled Matrix Multiplication (via compiler optimization).
- **Custom Layers**: `Conv2d`, `Linear`, `ReLU`, `MaxPool2d` implemented from scratch.
