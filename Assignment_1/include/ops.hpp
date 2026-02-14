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

// Convolution (Returns {output, col_matrix} for backward)
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

// Conv2d Backward
Tensor conv2d_backward_input(const Tensor& grad_output, const Tensor& kernel, int stride, int padding, const Tensor& input_shape_ref); 
Tensor conv2d_backward_kernel(const Tensor& grad_output, const Tensor& input, const Tensor& col_matrix, int stride, int padding); 

// MaxPool Backward
Tensor maxpool2d_backward(const Tensor& grad_output, const Tensor& input_shape_ref, const Tensor& indices);

// Loss
Tensor cross_entropy(const Tensor& input, const Tensor& target); 
Tensor cross_entropy_backward(const Tensor& input, const Tensor& target);

// Optimizers
void sgd_step(Tensor& param, const Tensor& grad, float lr);
void adam_step(Tensor& param, const Tensor& grad, Tensor& m, Tensor& v, float lr, float beta1, float beta2, float eps, int t);
void adam_step_wd(Tensor& param, const Tensor& grad, Tensor& m, Tensor& v, float lr, float beta1, float beta2, float eps, int t, float weight_decay);

// Dropout: returns {output, mask}
std::pair<Tensor, Tensor> dropout(const Tensor& input, float p, bool training);
// Dropout backward: grad_output * mask
Tensor dropout_backward(const Tensor& grad_output, const Tensor& mask);

// Global Average Pooling: [N, C, H, W] -> [N, C]
Tensor global_avg_pool2d(const Tensor& input);
Tensor global_avg_pool2d_backward(const Tensor& grad_output, int H, int W);

// Data Augmentation: Random horizontal flip for [N, C, H, W] tensors
Tensor random_horizontal_flip(const Tensor& input, float p);
// Data Augmentation: Pad image by `pad` pixels, random-crop back with probability `p` per image
Tensor random_crop_with_padding(const Tensor& input, int pad, float p);


Tensor load_image(const std::string& path);
Tensor load_image_batch(const std::vector<std::string>& paths, int C, int H, int W); 

// For accuracy: input [B, C]. Returns vector of ints
std::vector<int> argmax(const Tensor& input, int axis);

Tensor random_uniform(const std::vector<int>& shape, float min_val, float max_val);
Tensor zeros(const std::vector<int>& shape);
Tensor ones(const std::vector<int>& shape);

}
