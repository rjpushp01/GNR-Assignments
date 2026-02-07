import pandas as pd
import matplotlib.pyplot as plt
import argparse

def plot_metrics(log_file):
    try:
        df = pd.read_csv(log_file)
    except FileNotFoundError:
        print(f"Error: {log_file} not found. Run training first.")
        return

    import os
    
    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(df['Epoch'], df['Train Loss'], label='Train Loss')
    plt.plot(df['Epoch'], df['Val Loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Save with unique name in results directory
    if not os.path.exists("results"):
        os.makedirs("results")
        
    base_name = os.path.splitext(os.path.basename(log_file))[0]
    loss_png = f"results/loss_curve_{base_name}.png"
    plt.savefig(loss_png)
    print(f"Saved {loss_png}")
    
    # Plot Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(df['Epoch'], df['Train Acc'], label='Train Acc')
    plt.plot(df['Epoch'], df['Val Acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    acc_png = f"results/accuracy_curve_{base_name}.png"
    plt.savefig(acc_png)
    print(f"Saved {acc_png}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_file', type=str, default='training_log.csv', help='Path to training log csv')
    args = parser.parse_args()
    plot_metrics(args.log_file)
