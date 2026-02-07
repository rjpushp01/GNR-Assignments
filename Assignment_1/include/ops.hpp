#pragma once
#include "tensor.hpp"
#include <vector>
#include <utility>

namespace ops {

// Matrix Multiplication
// A: [M, K], B: [K, N] -> [M, N]
Tensor matmul(const Tensor& a, const Tensor& b);

// Transpose (2D)
Tensor transpose(const Tensor& input);

// Convolution (2D via Im2Col)
// Returns {output, col_matrix} for backward
std::pair<Tensor, Tensor> conv2d(const Tensor& input, const Tensor& kernel, int stride, int padding);

// Max Pool 2D
// Returns {output, indices} for backward
std::pair<Tensor, Tensor> maxpool2d(const Tensor& input, int kernel_size, int stride);

// Activation
Tensor relu(const Tensor& input);
Tensor relu_backward(const Tensor& grad_output, const Tensor& input);

// Backward Ops
// MatMul Grad
Tensor matmul_backward_input(const Tensor& grad_output, const Tensor& other); // dA = dC * B^T
Tensor matmul_backward_other(const Tensor& grad_output, const Tensor& input); // dB = A^T * dC

// Conv2d Grad - Optimized using saved Col matrix from forward
// dL/dW = Col^T * dL/dY_reshaped
// dL/dInput = Col2Im(dL/dY_reshaped * W)
Tensor conv2d_backward_input(const Tensor& grad_output, const Tensor& kernel, int stride, int padding, const Tensor& input_shape_ref); 
Tensor conv2d_backward_kernel(const Tensor& grad_output, const Tensor& input, const Tensor& col_matrix, int stride, int padding); 

// MaxPool Grad - Optimized using saved indices from forward
Tensor maxpool2d_backward(const Tensor& grad_output, const Tensor& input_shape_ref, const Tensor& indices);

// Loss
Tensor cross_entropy(const Tensor& input, const Tensor& target); 
Tensor cross_entropy_backward(const Tensor& input, const Tensor& target);

// Optimizers
void sgd_step(Tensor& param, const Tensor& grad, float lr);
void adam_step(Tensor& param, const Tensor& grad, Tensor& m, Tensor& v, float lr, float beta1, float beta2, float eps, int t);

}
