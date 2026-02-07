# Deep Learning Framework Instructions

## Project Structure
- **src/**: C++ source code for the backend (`my_backend`) containing core tensor operations.
- **python/**: Python wrapper (`my_framework`) that interfaces with the C++ backend.
- **tests/**: Unit tests for verifying framework components (e.g., `test_forward.py` checks tensor operations).
- **results/**: Directory where all training logs, saved models, plots, and test results are stored.

## Prerequisites
- **Conda** (for environment management)
- **C++ Compiler** (GCC/Clang for Linux/Mac, MSVC for Windows)
- **CMake** (v3.10+)
- **Make**

## Setup
1. **Create Environment**:
   ```bash
   conda env create -f environment.yml
   conda activate gnr638_ass
   ```
2. **Build C++ Backend**:
   ```bash
   # Ensure you are in the gnr638_ass environment
   cmake -S . -B build -DPYTHON_EXECUTABLE=$(which python)
   cmake --build build
   ```

## Usage

### 1. Training
Run the training script for MNIST or CIFAR. Logs and models will be saved to `results/`.
```bash
# Train on MNIST
python train.py --dataset mnist --data_path data_1 --epochs 5 --lr 0.001

# Train on CIFAR
python train.py --dataset cifar --data_path data_2 --epochs 10 --lr 0.001
```
**Outputs**:
- `results/mnist_model.pkl` (or `cifar_model.pkl`)
- `results/training_log_mnist.csv` (or `training_log_cifar.csv`)

### 2. Testing
Evaluate the trained model on the test set. Results are appended to `results/test_results.txt`.
```bash
# Test MNIST
python test.py --dataset mnist --data_path data_1 --model_path results/mnist_model.pkl

# Test CIFAR
python test.py --dataset cifar --data_path data_2 --model_path results/cifar_model.pkl
```

### 3. Plotting
Generate loss and accuracy curves from the training logs. Plots are saved to `results/`.
```bash
# Plot MNIST Metrics
python plot_metrics.py --log_file results/training_log_mnist.csv

# Plot CIFAR Metrics
python plot_metrics.py --log_file results/training_log_cifar.csv
```
**Outputs**:
- `results/loss_curve_training_log_mnist.png`
- `results/accuracy_curve_training_log_mnist.png`

## Analyzing Results
All artifacts are in the `results/` folder.
- Check `test_results.txt` for a summary of model accuracy.
- View the `.png` plots to visualize training progress.

## Run Unit Tests
The `tests/` directory contains unit tests to verify the correctness of the framework's core components (e.g., convolution, matrix multiplication).
```bash
python -m unittest discover tests
```
