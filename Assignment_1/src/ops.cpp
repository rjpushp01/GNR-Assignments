#include "ops.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>

namespace ops {

// Utility: Im2Col
void im2col(const std::vector<float>& input_data, std::vector<float>& col_data,
            int N, int C, int H, int W, int KH, int KW, int stride, int padding,
            int H_out, int W_out) {
            
    int col_width = C * KH * KW; // K_col
    // Parallelize batch loop
    #pragma omp parallel for 
    for (int n = 0; n < N; ++n) {
        for (int h_out = 0; h_out < H_out; ++h_out) {
            for (int w_out = 0; w_out < W_out; ++w_out) {
                int col_row_idx = n * (H_out * W_out) + h_out * W_out + w_out; // M_col index
                
                int col_idx = 0;
                for (int c = 0; c < C; ++c) {
                    for (int kh = 0; kh < KH; ++kh) {
                        for (int kw = 0; kw < KW; ++kw) {
                            
                            int h_in = h_out * stride - padding + kh;
                            int w_in = w_out * stride - padding + kw;
                            
                            float val = 0.0f;
                            if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                                val = input_data[n*(C*H*W) + c*(H*W) + h_in*W + w_in];
                            }
                            col_data[col_row_idx * col_width + col_idx] = val;
                            col_idx++;
                        }
                    }
                }
            }
        }
    }
}

// Utility: Col2Im
void col2im(const std::vector<float>& col_data, std::vector<float>& input_diff,
            int N, int C, int H, int W, int KH, int KW, int stride, int padding,
            int H_out, int W_out) {
            
    int col_width = C * KH * KW;
    int input_hw = H * W;
    
    // Initialize input_diff to 0 (done in caller)
    
    for (int n = 0; n < N; ++n) {
        for (int h_out = 0; h_out < H_out; ++h_out) {
            for (int w_out = 0; w_out < W_out; ++w_out) {
                int col_row_idx = n * (H_out * W_out) + h_out * W_out + w_out;
                
                int col_idx = 0;
                for (int c = 0; c < C; ++c) {
                    for (int kh = 0; kh < KH; ++kh) {
                        for (int kw = 0; kw < KW; ++kw) {
                            
                            int h_in = h_out * stride - padding + kh;
                            int w_in = w_out * stride - padding + kw;
                            
                            if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                                float val = col_data[col_row_idx * col_width + col_idx];
                                input_diff[n*(C*input_hw) + c*input_hw + h_in*W + w_in] += val;
                            }
                            col_idx++;
                        }
                    }
                }
            }
        }
    }
}


Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.shape.size() != 2 || b.shape.size() != 2) throw std::runtime_error("MatMul requires 2D");
    int M = a.shape[0];
    int K = a.shape[1];
    int N = b.shape[1];
    if (K != b.shape[0]) throw std::runtime_error("MatMul mismatch");

    Tensor C({M, N}, std::vector<float>(M*N, 0.0f));
    
    // Tiled Matrix Multiplication Cache Optimization
    constexpr int BS = 64; 
    for (int i0 = 0; i0 < M; i0 += BS)
        for (int k0 = 0; k0 < K; k0 += BS)
             for (int j0 = 0; j0 < N; j0 += BS)
                 for (int i = i0; i < std::min(i0+BS, M); ++i)
                     for (int k = k0; k < std::min(k0+BS, K); ++k) {
                         float val_a = a.data[i*K + k];
                         for (int j = j0; j < std::min(j0+BS, N); ++j)
                             C.data[i*N + j] += val_a * b.data[k*N + j];
                     }
    return C;
}

Tensor transpose(const Tensor& input) {
    if (input.shape.size() != 2) throw std::runtime_error("Transpose requires 2D");
    int M = input.shape[0];
    int N = input.shape[1];
    std::vector<float> res(M * N);
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            res[j * M + i] = input.data[i * N + j];
    return Tensor({N, M}, res);
}

std::pair<Tensor, Tensor> conv2d(const Tensor& input, const Tensor& kernel, int stride, int padding) {
    int N = input.shape[0];
    int C = input.shape[1];
    int H = input.shape[2];
    int W = input.shape[3];
    int OutC = kernel.shape[0];
    int KH = kernel.shape[2];
    int KW = kernel.shape[3];

    int H_out = (H + 2 * padding - KH) / stride + 1;
    int W_out = (W + 2 * padding - KW) / stride + 1;
    
    // Im2Col
    int M_col = N * H_out * W_out;
    int K_col = C * KH * KW;
    std::vector<float> col_data(M_col * K_col);
    im2col(input.data, col_data, N, C, H, W, KH, KW, stride, padding, H_out, W_out);
    Tensor col_tensor({M_col, K_col}, col_data);

    // Prepare Kernel as Matrix [OutC, K_col]
    // Kernel layout is [OutC, InC, KH, KW], which flattens directly to Row-Major [OutC, K_col].
    // We want Output = Col * Kernel^T -> [M_col, OutC]
    // So if K is [OutC, K_col], K^T is [K_col, OutC].
    // We construct MatMul(Col, K^T).
    
    // Step 1: Transpose Kernel Matrix
    // Treat as 2D [OutC, K_col]
    Tensor kernel_flat({OutC, K_col}, kernel.data); 
    Tensor kernel_t = transpose(kernel_flat); // [K_col, OutC]
    
    // Step 2: MatMul
    Tensor result_mm = matmul(col_tensor, kernel_t); // [M_col, OutC]

    // Step 3: Reshape/Permute to [N, OutC, H_out, W_out]
    // result_mm is [N * H_out * W_out, OutC] -> corresponds to [N, H_out, W_out, OutC].
    // Needs permute from (N, H, W, C) -> (N, C, H, W)
    std::vector<float> out_data(result_mm.size());
    const auto& res_d = result_mm.data;
    
    for(int n=0; n<N; ++n)
        for(int h=0; h<H_out; ++h)
            for(int w=0; w<W_out; ++w)
                for(int c=0; c<OutC; ++c) {
                    int src = n*(H_out*W_out*OutC) + h*(W_out*OutC) + w*OutC + c;
                    int dst = n*(OutC*H_out*W_out) + c*(H_out*W_out) + h*W_out + w;
                    out_data[dst] = res_d[src];
                }

    return {Tensor({N, OutC, H_out, W_out}, out_data), col_tensor};
}

std::pair<Tensor, Tensor> maxpool2d(const Tensor& input, int k, int stride) {
    int N = input.shape[0];
    int C = input.shape[1];
    int H = input.shape[2];
    int W = input.shape[3]; 

    int H_out = (H - k) / stride + 1;
    int W_out = (W - k) / stride + 1;

    Tensor out({N, C, H_out, W_out}, std::vector<float>(N*C*H_out*W_out));
    Tensor indices({N, C, H_out, W_out}, std::vector<float>(N*C*H_out*W_out)); // Store flat index as float

    for (int n=0; n<N; ++n)
        for (int c=0; c<C; ++c)
            for (int h=0; h<H_out; ++h)
                for (int w=0; w<W_out; ++w) {
                    
                    float max_val = -1e30f;
                    int max_idx = -1;
                    
                    for (int kh=0; kh<k; ++kh)
                        for (int kw=0; kw<k; ++kw) {
                            int cur_h = h*stride + kh;
                            int cur_w = w*stride + kw;
                            int idx = n*(C*H*W) + c*(H*W) + cur_h*W + cur_w;
                            if (input.data[idx] > max_val) {
                                max_val = input.data[idx];
                                max_idx = idx;
                            }
                        }
                    int out_idx = n*(C*H_out*W_out) + c*(H_out*W_out) + h*W_out + w;
                    out.data[out_idx] = max_val;
                    indices.data[out_idx] = (float)max_idx;
                }
    return {out, indices};
}

Tensor relu(const Tensor& x) {
    Tensor y(x.shape, x.data);
    for (auto& v : y.data) v = std::max(0.0f, v);
    return y;
}

Tensor relu_backward(const Tensor& grad, const Tensor& x) {
    Tensor dx(grad.shape, std::vector<float>(grad.size()));
    for (size_t i = 0; i < x.size(); ++i)
        dx.data[i] = x.data[i] > 0 ? grad.data[i] : 0.0f;
    return dx;
}

Tensor matmul_backward_input(const Tensor& grad_output, const Tensor& other) {
    return matmul(grad_output, transpose(other));
}

Tensor matmul_backward_other(const Tensor& grad_output, const Tensor& input) {
    return matmul(transpose(input), grad_output);
}

// Optimized Conv2d Backward using saved Col Tensor
Tensor conv2d_backward_input(const Tensor& grad_output, const Tensor& kernel, int stride, int padding, const Tensor& input_shape_ref) {
    // dL/dX = Col2Im( dL/dY * W )
    // grad_output: [N, OutC, H_out, W_out]
    // kernel: [OutC, InC, KH, KW]
    
    int N = input_shape_ref.shape[0];
    int C = input_shape_ref.shape[1];
    int H = input_shape_ref.shape[2];
    int W = input_shape_ref.shape[3];
    int OutC = grad_output.shape[1];
    int H_out = grad_output.shape[2];
    int W_out = grad_output.shape[3];
    int KH = kernel.shape[2];
    int KW = kernel.shape[3];
    int K_col = C * KH * KW;

    // Reshape grad to [N, H_out, W_out, OutC] -> [M_col, OutC]
    // Permute (N, C, H, W) -> (N, H, W, C)
    std::vector<float> grad_flat(grad_output.size());
    for(int n=0; n<N; ++n)
        for(int h=0; h<H_out; ++h)
            for(int w=0; w<W_out; ++w)
                for(int c=0; c<OutC; ++c) {
                    grad_flat[n*(H_out*W_out*OutC) + h*(W_out*OutC) + w*OutC + c] = grad_output.data[n*(OutC*H_out*W_out) + c*(H_out*W_out) + h*W_out + w];
                }
    Tensor grad_mat({N*H_out*W_out, OutC}, grad_flat);
    
    // Kernel Matrix [OutC, K_col].
    // We need dCol = GradMat * KernelMat -> [M_col, OutC] * [OutC, K_col] -> [M_col, K_col]
    Tensor kernel_mat({OutC, K_col}, kernel.data); 
    
    Tensor dCol = matmul(grad_mat, kernel_mat); // [M_col, K_col]
    
    // Col2Im
    std::vector<float> dInput_data(N*C*H*W, 0.0f);
    col2im(dCol.data, dInput_data, N, C, H, W, KH, KW, stride, padding, H_out, W_out);
    
    return Tensor({N, C, H, W}, dInput_data);
}

Tensor conv2d_backward_kernel(const Tensor& grad_output, const Tensor& input, const Tensor& col_matrix, int stride, int padding) {
    // dL/dW = Col^T * dL/dY
    // col_matrix: [M_col, K_col] (Saved from Forward)
    // grad_output (reshaped as dY_mat): [M_col, OutC]
    
    int OutC = grad_output.shape[1];
    int N = grad_output.shape[0];
    int H_out = grad_output.shape[2];
    int W_out = grad_output.shape[3];
    int K_col = col_matrix.shape[1];

    // Reshape grad as usual
    std::vector<float> grad_flat(grad_output.size());
    for(int n=0; n<N; ++n)
        for(int h=0; h<H_out; ++h)
            for(int w=0; w<W_out; ++w)
                for(int c=0; c<OutC; ++c) {
                    grad_flat[n*(H_out*W_out*OutC) + h*(W_out*OutC) + w*OutC + c] = grad_output.data[n*(OutC*H_out*W_out) + c*(H_out*W_out) + h*W_out + w];
                }
    Tensor grad_mat({N*H_out*W_out, OutC}, grad_flat); // [M_col, OutC]
    
    // dW_trans = Col^T * Grad -> [K_col, M_col] * [M_col, OutC] -> [K_col, OutC]
    Tensor dW_trans = matmul(transpose(col_matrix), grad_mat);
    
    // dW = Transpose(dW_trans) -> [OutC, K_col].
    // Kernel is stored as [OutC, InC, KH, KW] which is [OutC, K_col] flat.
    Tensor dW = transpose(dW_trans); // [OutC, K_col] which matches flat memory layout of [OutC, InC, KH, KW]
    
    // Reshape metadata is handled by Python wrapper usually, or we return tensor with correct shape logic?
    // We return flat data basically, shape handled by caller/wrapper. But let's verify C++ shape.
    // We just return [OutC, InC, KH, KW] shape from python side logic?
    // My C++ conv_backward signature doesn't take 'kernel_size', so I can't reconstruct shape here easily inside C++ without extra args.
    // Wait, I updated signature to not take kernel size?
    // My header has: `conv2d_backward_kernel(grad, input_shape, col_matrix, ...)`
    // Actually, I just return the dW tensor. The Python wrapper knows the shape of W. 
    // Let's modify tensor.py to reshape it.
    // Or just return flat/matrix shape and let tensor wrapper raw-assign?
    return dW; 
}


Tensor maxpool2d_backward(const Tensor& grad_output, const Tensor& input_shape_ref, const Tensor& indices) {
    // indices stores flat index of max.
    // grad_output and indices have same shape [N, C, H_out, W_out]
    
    Tensor dx(input_shape_ref.shape, std::vector<float>(input_shape_ref.size(), 0.0f));
    
    for (size_t i = 0; i < grad_output.size(); ++i) {
        int idx = (int)indices.data[i];
        dx.data[idx] += grad_output.data[i];
    }
    return dx;
}

Tensor cross_entropy(const Tensor& logits, const Tensor& target) {
    int N = logits.shape[0];
    int C = logits.shape[1];
    float loss = 0.0f;
    for (int n=0;n<N;++n) {
        float m = -1e30f;
        for (int c=0;c<C;++c) m = std::max(m, logits.data[n*C+c]);
        float sum = 0.0f;
        for (int c=0;c<C;++c) sum += std::exp(logits.data[n*C+c] - m);
        int t = (int)target.data[n];
        loss += -logits.data[n*C+t] + m + std::log(sum);
    }
    return Tensor({1}, {loss / N});
}

Tensor cross_entropy_backward(const Tensor& logits, const Tensor& target) {
    int N = logits.shape[0];
    int C = logits.shape[1];
    Tensor grad(logits.shape, std::vector<float>(N*C));
    for (int n=0;n<N;++n) {
        float m = -1e30f;
        for (int c=0;c<C;++c) m = std::max(m, logits.data[n*C+c]);
        float sum = 0.0f;
        for (int c=0;c<C;++c) sum += std::exp(logits.data[n*C+c] - m);
        int t = (int)target.data[n];
        for (int c=0;c<C;++c) {
            float s = std::exp(logits.data[n*C+c] - m) / sum;
            grad.data[n*C+c] = (s - (c == t)) / N;
        }
    }
    return grad;
}

void sgd_step(Tensor& param, const Tensor& grad, float lr) {
    if (param.size() != grad.size()) throw std::runtime_error("SGD size mismatch");
    for (size_t i = 0; i < param.size(); ++i) param.data[i] -= lr * grad.data[i];
}

void adam_step(Tensor& param, const Tensor& grad, Tensor& m, Tensor& v, float lr, float beta1, float beta2, float eps, int t) {
    if (param.size() != grad.size()) throw std::runtime_error("Adam size mismatch");
    float bc1 = 1.0f - std::pow(beta1, t);
    float bc2 = 1.0f - std::pow(beta2, t);
    for (size_t i = 0; i < param.size(); ++i) {
        float g = grad.data[i];
        m.data[i] = beta1 * m.data[i] + (1.0f - beta1) * g;
        v.data[i] = beta2 * v.data[i] + (1.0f - beta2) * g * g;
        float m_hat = m.data[i] / bc1;
        float v_hat = v.data[i] / bc2;
        param.data[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
    }
}

}
