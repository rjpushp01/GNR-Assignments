import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
python_path = os.path.join(script_dir, 'python')
if python_path not in sys.path:
    sys.path.insert(0, python_path)

import argparse
import time
import pickle
import my_backend as mb
from my_framework.models import MNIST_Model, CIFAR_Model
from my_framework.data import DataLoader
from my_framework.model_utils import print_model_summary

def accuracy(outputs, labels):
    preds = mb.ops.argmax(outputs.data, 1)
    targets = labels.to_list()
    
    correct = 0
    for p, t in zip(preds, targets):
        if p == int(t):
            correct += 1
    return correct / len(targets)

def load_model(path, model):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file {path} not found.")
        
    print(f"Loading model from {path}...")
    with open(path, 'rb') as f:
        params = pickle.load(f)
        
    for name, layer in model.__dict__.items():
        if hasattr(layer, 'weight') and w_key in params:
            p = params[w_key]
            layer.weight.data = mb.Tensor(p['shape'], p['data'])
            
        b_key = name + '.bias'
        if hasattr(layer, 'bias') and b_key in params:
            p = params[b_key]
            layer.bias.data = mb.Tensor(p['shape'], p['data'])
            
    print("Model loaded successfully.")

def test(args):
    print(f"  TESTING DATASET: {args.dataset.upper()}")

    print(f"Loading test data from '{args.data_path}' ...")
    load_start = time.time()
    test_loader = DataLoader(args.data_path, batch_size=args.batch_size, shuffle=False, mode='test')
    load_time = time.time() - load_start
    print(f"  Test data loaded in {load_time:.2f}s")

    if args.dataset == 'mnist':
        model = MNIST_Model()
        input_shape = (1, 32, 32)
    else:
        model = CIFAR_Model()
        input_shape = (3, 32, 32)

    load_model(args.model_path, model)
    model.eval()

    print("\nModel Summary:")
    model_summary = print_model_summary(model, input_shape)

    total_acc = 0.0
    num_batches = 0
    total_samples = 0
    
    print("\nStarting evaluation...")
    eval_start = time.time()
    
    for images, labels in test_loader:
        outputs = model(images)
        
        acc = accuracy(outputs, labels)
        
        total_acc += acc
        num_batches += 1
        total_samples += len(labels.to_list())
        
        if num_batches % 10 == 0:
            print(f"  Batch {num_batches}: Acc = {acc:.4f}")

    eval_time = time.time() - eval_start
    avg_acc = total_acc / num_batches if num_batches > 0 else 0

    print("\n" + "=" * 70)
    print("  TEST RESULTS")
    print("=" * 70)
    print(f"  Dataset           : {args.dataset.upper()}")
    print(f"  Model path        : {args.model_path}")
    print(f"  Test samples      : {test_loader.num_samples}")
    print(f"  Test Accuracy     : {avg_acc:.4f}  ({avg_acc*100:.2f}%)")
    print(f"  Data loading time : {load_time:.2f}s")
    print(f"  Evaluation time   : {eval_time:.2f}s")
    print(f"  Trainable params  : {model_summary['total_params']:,}")
    print(f"  MACs / forward    : {model_summary['total_macs']:,}")
    print(f"  FLOPs / forward   : {model_summary['total_flops']:,}")
    print("=" * 70)

    os.makedirs("results/test_results", exist_ok=True)
    result_file = f"results/test_results/{args.dataset}_test_results.txt"
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(result_file, "w") as f:
        f.write(f"{'=' * 60}\n")
        f.write(f"Evaluation — {args.dataset.upper()}\n")
        f.write(f"Run timestamp     : {timestamp}\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"Model path        : {args.model_path}\n")
        f.write(f"Test samples      : {test_loader.num_samples}\n")
        f.write(f"Test Accuracy     : {avg_acc:.4f}  ({avg_acc*100:.2f}%)\n")
        f.write(f"Data loading time : {load_time:.2f}s\n")
        f.write(f"Evaluation time   : {eval_time:.2f}s\n")
        f.write(f"Trainable params  : {model_summary['total_params']:,}\n")
        f.write(f"MACs / forward    : {model_summary['total_macs']:,}\n")
        f.write(f"FLOPs / forward   : {model_summary['total_flops']:,}\n")
    print(f"\nResults saved to {result_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['mnist', 'cifar'])
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    
    args = parser.parse_args()
    test(args)
