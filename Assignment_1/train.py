import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
python_path = os.path.join(script_dir, 'python')
if python_path not in sys.path:
    sys.path.insert(0, python_path)

# Verify my_backend is importable
try:
    import my_backend
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import 'my_backend' from {python_path}")
    print(f"Current sys.path: {sys.path}")
    print("Ensure you have built the backend with: cmake -S . -B build && cmake --build build")
    raise e

import argparse
import time
import pickle
import csv
import math
import resource  # For memory usage (standard library)

# Custom Imports
from my_framework.data import DataLoader
from my_framework.models import MNIST_Model, CIFAR_Model, CrossEntropyLoss
from my_framework.optim import Adam
from my_framework.tensor import Tensor
import my_backend as mb # Access ops
from my_framework.model_utils import print_model_summary, count_parameters


def accuracy(outputs, labels):
    preds = mb.ops.argmax(outputs.data, 1)
    
    targets = labels.to_list()
    
    correct = 0
    for p, t in zip(preds, targets):
        if p == int(t):
            correct += 1
    return correct / len(targets)


def save_model(model, path):
    params = {}
    for name, value in model.__dict__.items():
         if hasattr(value, 'weight'):
             params[name + '.weight'] = {'data': value.weight.to_list(), 'shape': value.weight.shape}
         if hasattr(value, 'bias') and value.bias is not None:
             params[name + '.bias'] = {'data': value.bias.to_list(), 'shape': value.bias.shape}

    with open(path, 'wb') as f:
        pickle.dump(params, f)
    print(f"Model saved to {path}")


def validate(model, val_loader, criterion):
    val_loss = 0.0
    val_acc = 0.0
    num_batches = 0

    for images, labels in val_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        acc = accuracy(outputs, labels)

        val_loss += loss.item()
        val_acc += acc
        num_batches += 1

    avg_loss = val_loss / num_batches if num_batches > 0 else 0
    avg_acc = val_acc / num_batches if num_batches > 0 else 0
    return avg_loss, avg_acc


def get_memory_mb():
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_maxrss / 1024 
    except Exception:
        return 0.0


def train(args):
    # Ensure results subdirectories exist
    for sub in ["results/models", "results/training_logs", "results/test_results", "results/plots"]:
        os.makedirs(sub, exist_ok=True)

    print(f"  DATASET: {args.dataset.upper()}")
    
    print(f"\nLoading training data from '{args.data_path}' ...")
    load_start = time.time()
    train_loader = DataLoader(args.data_path, batch_size=args.batch_size, shuffle=True, mode='train')
    train_load_time = time.time() - load_start
    print(f"  Training data loaded in {train_load_time:.2f}s")

    print(f"Loading validation data from '{args.data_path}' ...")
    load_start = time.time()
    val_loader = DataLoader(args.data_path, batch_size=args.batch_size, shuffle=False, mode='val')
    val_load_time = time.time() - load_start
    print(f"  Validation data loaded in {val_load_time:.2f}s")

    total_load_time = train_load_time + val_load_time
    print(f"  Total dataset loading time: {total_load_time:.2f}s")

    if args.dataset == 'mnist':
        model = MNIST_Model()
        input_shape = (1, 32, 32)
    else:
        model = CIFAR_Model()
        input_shape = (3, 32, 32)

    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    summary_path = f"results/training_logs/model_summary_{args.dataset}.txt"
    model_summary = print_model_summary(model, input_shape, file=summary_path)

    total_params = model_summary["total_params"]
    total_macs = model_summary["total_macs"]
    total_flops = model_summary["total_flops"]

    log_filename = f"results/training_logs/training_log_{args.dataset}.csv"
    log_file = open(log_filename, "w", newline='')
    writer = csv.writer(log_file)
    writer.writerow([
        "Epoch", "Train Loss", "Train Acc", "Val Loss", "Val Acc",
        "Epoch Time (s)", "Cumulative Time (s)", "Avg Batch Time (s)",
        "Num Batches", "Learning Rate", "Memory (MB)"
    ])

    print("\n" + "=" * 70)
    print("  TRAINING CONFIGURATION")
    print("=" * 70)
    print(f"  Dataset           : {args.dataset.upper()}")
    print(f"  Data path         : {args.data_path}")
    print(f"  Epochs            : {args.epochs}")
    print(f"  Batch size        : {args.batch_size}")
    print(f"  Dataset load time : {total_load_time:.2f}s")
    print(f"  Weight decay      : {args.weight_decay}")
    print("=" * 70)
    print()

    cumulative_time = 0.0
    best_val_acc = 0.0
    best_epoch = 0

    initial_lr = args.lr
    min_lr = initial_lr * 0.1
    total_epochs = args.epochs

    def cosine_lr(epoch, total_epochs, initial_lr, min_lr):
        """Cosine annealing: lr decays from initial_lr to min_lr over total_epochs."""
        return min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * epoch / total_epochs))

    print(f"Starting training for {args.epochs} epochs...")
    print(f"  LR Schedule: Cosine Annealing ({initial_lr:.4f} → {min_lr:.6f})\n")

    for epoch in range(args.epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0
        batch_times = []

        # Update learning rate (cosine annealing)
        current_lr = cosine_lr(epoch, total_epochs, initial_lr, min_lr)
        optimizer.lr = current_lr

        for i, (images, labels) in enumerate(train_loader):
            batch_start = time.time()

            # Forward (training mode)
            model.train()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Update
            optimizer.step()

            # Metrics
            acc = accuracy(outputs, labels)
            epoch_loss += loss.item()
            epoch_acc += acc
            num_batches += 1
            batch_times.append(time.time() - batch_start)

            if i % 10 == 0:
                print(f"  Epoch [{epoch+1}/{args.epochs}], Step [{i}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}, Acc: {acc:.4f}, "
                      f"Batch: {batch_times[-1]:.3f}s")

        avg_loss = epoch_loss / num_batches
        avg_acc = epoch_acc / num_batches
        avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0

        val_start = time.time()
        model.eval()
        val_loss, val_acc = validate(model, val_loader, criterion)
        val_time = time.time() - val_start

        epoch_duration = time.time() - epoch_start
        cumulative_time += epoch_duration
        mem_mb = get_memory_mb()

        # Track best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1

        print(f"\n{'─' * 70}")
        print(f"  Epoch {epoch+1}/{args.epochs} Summary:")
        print(f"    Learning Rate  : {current_lr:.6f}")
        print(f"    Train Loss     : {avg_loss:.4f}")
        print(f"    Train Accuracy : {avg_acc:.4f}")
        print(f"    Val Loss       : {val_loss:.4f}")
        print(f"    Val Accuracy   : {val_acc:.4f}")
        print(f"    Epoch Time     : {epoch_duration:.2f}s")
        print(f"    Avg Batch Time : {avg_batch_time:.4f}s")
        print(f"    Memory (RSS)   : {mem_mb:.1f} MB")
        print(f"{'─' * 70}\n")

        writer.writerow([
            epoch + 1, avg_loss, avg_acc, val_loss, val_acc,
            epoch_duration, cumulative_time, avg_batch_time,
            num_batches, current_lr, mem_mb
        ])
        log_file.flush()

    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Total epochs      : {args.epochs}")
    print(f"  Total time        : {cumulative_time:.2f}s")
    print(f"  Best Val Accuracy : {best_val_acc:.4f}  (Epoch {best_epoch})")
    print("=" * 70)

    save_path = f"results/models/{args.dataset}_model.pkl"
    save_model(model, save_path)
    log_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CNN model using custom framework")
    parser.add_argument('--dataset', type=str, required=True, choices=['mnist', 'cifar'])
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0, help='AdamW weight decay (L2 reg)')
    
    args = parser.parse_args()
    train(args)
