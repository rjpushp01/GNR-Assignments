import sys
import os
sys.path.append(os.path.abspath('python'))
import argparse
import numpy as np
import pickle
import os

from my_framework.data import DataLoader
from my_framework.models import MNIST_Model, CIFAR_Model
from my_framework.tensor import Tensor

def load_model(path):
    with open(path, 'rb') as f:
        params = pickle.load(f)
    return params

def accuracy(outputs, labels):
    preds = np.argmax(outputs.numpy(), axis=1)
    targets = labels.numpy().astype(int)
    return np.mean(preds == targets)

def test(args):
    print(f"Loading data from {args.data_path} for testing...")
    test_loader = DataLoader(args.data_path, batch_size=args.batch_size, shuffle=False, mode='test')
    
    if args.dataset == 'mnist':
        model = MNIST_Model()
    else:
        model = CIFAR_Model()
        
    print(f"Loading model from {args.model_path}...")
    params = load_model(args.model_path)
    
    import my_backend as mb 
    
    # Load parameters into model
    for name, value in model.__dict__.items():
        if hasattr(value, 'weight') and (name + '.weight') in params:
             # Convert numpy array back to C++ Tensor
             param_data = params[name + '.weight']
             # param_data is (Out, In, H, W) for Conv or (In, Out) for Linear?
             # Wait, Tensor constructor takes (shape, flat_data)
             # The saved params are numpy arrays.
             shape = list(param_data.shape)
             flat_data = param_data.flatten().tolist()
             value.weight.data = mb.Tensor(shape, flat_data)
             
        if hasattr(value, 'bias') and (name + '.bias') in params:
             param_data = params[name + '.bias']
             shape = list(param_data.shape)
             flat_data = param_data.flatten().tolist()
             value.bias.data = mb.Tensor(shape, flat_data)
             
    print("Starting Evaluation...")
    total_acc = 0.0
    num_batches = 0
    
    for images, labels in test_loader:
        outputs = model(images)
        acc = accuracy(outputs, labels)
        total_acc += acc
        num_batches += 1
        
    avg_acc = total_acc / num_batches if num_batches > 0 else 0
    print(f"Test Accuracy: {avg_acc:.4f}")
    
    # Save results to file
    if not os.path.exists("results"):
        os.makedirs("results")
        
    with open("results/test_results.txt", "a") as f:
        f.write(f"Dataset: {args.dataset}, Model Path: {args.model_path}, Accuracy: {avg_acc:.4f}\n")
    print("Result saved to results/test_results.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['mnist', 'cifar'], help='Dataset type')
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset root')
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved model pickle')
    parser.add_argument('--batch_size', type=int, default=32)
    
    args = parser.parse_args()
    test(args)
