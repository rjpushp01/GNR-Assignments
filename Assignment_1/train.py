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
import numpy as np
import pickle
import csv

from my_framework.data import DataLoader
from my_framework.models import MNIST_Model, CIFAR_Model, CrossEntropyLoss
from my_framework.optim import Adam
from my_framework.tensor import Tensor

def accuracy(outputs, labels):
    # outputs: [B, C], labels: [B]
    # argmax of outputs
    preds = np.argmax(outputs.numpy(), axis=1)
    targets = labels.numpy().astype(int)
    return np.mean(preds == targets)

def save_model(model, path):
    params = {}
    for name, value in model.__dict__.items():
         # Basic serialization: save numpy arrays of parameters
         if hasattr(value, 'weight'): # Layer with weight
             params[name + '.weight'] = value.weight.numpy()
         if hasattr(value, 'bias') and value.bias is not None:
             params[name + '.bias'] = value.bias.numpy()
    
    with open(path, 'wb') as f:
        pickle.dump(params, f)
    print(f"Model saved to {path}")

def validate(model, val_loader, criterion):
    val_loss = 0.0
    val_acc = 0.0
    num_batches = 0
    
    # Disable gradient tracking if possible (here we just don't call backward)
    for images, labels in val_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        acc = accuracy(outputs, labels)
        
        val_loss += loss.numpy().item()
        val_acc += acc
        num_batches += 1
        
    avg_loss = val_loss / num_batches if num_batches > 0 else 0
    avg_acc = val_acc / num_batches if num_batches > 0 else 0
    return avg_loss, avg_acc

def train(args):
    # Ensure results directory
    if not os.path.exists("results"):
        os.makedirs("results")

    # Create log file
    log_filename = f"results/training_log_{args.dataset}.csv"
    log_file = open(log_filename, "w", newline='')
    writer = csv.writer(log_file)
    writer.writerow(["Epoch", "Train Loss", "Train Acc", "Val Loss", "Val Acc", "Time (s)"])

    if args.dataset == 'mnist':
        print("Initializing MNIST Training...")
        train_loader = DataLoader(args.data_path, batch_size=args.batch_size, shuffle=True, mode='train')
        val_loader = DataLoader(args.data_path, batch_size=args.batch_size, shuffle=False, mode='val')
        model = MNIST_Model()
    else:
        print("Initializing CIFAR Training...")
        train_loader = DataLoader(args.data_path, batch_size=args.batch_size, shuffle=True, mode='train')
        val_loader = DataLoader(args.data_path, batch_size=args.batch_size, shuffle=False, mode='val')
        model = CIFAR_Model()
        
    criterion = CrossEntropyLoss()
    # Use Adam Optimizer
    optimizer = Adam(model.parameters(), lr=args.lr)
    
    print(f"Start Training for {args.epochs} epochs")
    
    for epoch in range(args.epochs):
        start_time = time.time()
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0
        
        for i, (images, labels) in enumerate(train_loader):
            # Forward
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # Update
            optimizer.step()
            
            # Metrics
            acc = accuracy(outputs, labels)
            epoch_loss += loss.numpy().item() # Scalar
            epoch_acc += acc
            num_batches += 1
            
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.numpy().item():.4f}, Acc: {acc:.4f}")
        
        avg_loss = epoch_loss / num_batches
        avg_acc = epoch_acc / num_batches
        
        # Validation
        print("Validating...")
        val_loss, val_acc = validate(model, val_loader, criterion)
        
        duration = time.time() - start_time
        
        print(f"End of Epoch {epoch+1}: Train Loss: {avg_loss:.4f}, Train Acc: {avg_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Time: {duration:.2f}s")
        
        writer.writerow([epoch+1, avg_loss, avg_acc, val_loss, val_acc, duration])
        log_file.flush()
        
    # Save Model
    save_path = f"results/{args.dataset}_model.pkl"
    save_model(model, save_path)
    log_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['mnist', 'cifar'], help='Dataset type')
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset root')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    
    args = parser.parse_args()
    train(args)
