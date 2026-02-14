#include "ops.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>
#include <omp.h> // Ensure OpenMP is available if using #pragma omp

namespace ops {


void im2col(const std::vector<float>& input_data, std::vector<float>& col_data,
            int N, int C, int H, int W, int KH, int KW, int stride, int padding,
            int H_out, int W_out) {
            
    int col_width = C * KH * KW; 
 
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


void col2im(const std::vector<float>& col_data, std::vector<float>& input_diff,
            int N, int C, int H, int W, int KH, int KW, int stride, int padding,
            int H_out, int W_out) {
            
    int col_width = C * KH * KW;
    int input_hw = H * W;
            
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
    Tensor kernel_flat({OutC, K_col}, kernel.data); 
    Tensor kernel_t = transpose(kernel_flat); // [K_col, OutC]
    
    // MatMul
    Tensor result_mm = matmul(col_tensor, kernel_t); // [M_col, OutC]

    // Reshape/Permute to [N, OutC, H_out, W_out]
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
    Tensor indices({N, C, H_out, W_out}, std::vector<float>(N*C*H_out*W_out));

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

Tensor conv2d_backward_input(const Tensor& grad_output, const Tensor& kernel, int stride, int padding, const Tensor& input_shape_ref) {
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

    std::vector<float> grad_flat(grad_output.size());
    for(int n=0; n<N; ++n)
        for(int h=0; h<H_out; ++h)
            for(int w=0; w<W_out; ++w)
                for(int c=0; c<OutC; ++c) {
                    grad_flat[n*(H_out*W_out*OutC) + h*(W_out*OutC) + w*OutC + c] = grad_output.data[n*(OutC*H_out*W_out) + c*(H_out*W_out) + h*W_out + w];
                }
    Tensor grad_mat({N*H_out*W_out, OutC}, grad_flat);
    
    Tensor kernel_mat({OutC, K_col}, kernel.data); 
    Tensor dCol = matmul(grad_mat, kernel_mat); 
    
    std::vector<float> dInput_data(N*C*H*W, 0.0f);
    col2im(dCol.data, dInput_data, N, C, H, W, KH, KW, stride, padding, H_out, W_out);
    
    return Tensor({N, C, H, W}, dInput_data);
}

Tensor conv2d_backward_kernel(const Tensor& grad_output, const Tensor& input, const Tensor& col_matrix, int stride, int padding) {
    int OutC = grad_output.shape[1];
    int N = grad_output.shape[0];
    int H_out = grad_output.shape[2];
    int W_out = grad_output.shape[3];

    std::vector<float> grad_flat(grad_output.size());
    for(int n=0; n<N; ++n)
        for(int h=0; h<H_out; ++h)
            for(int w=0; w<W_out; ++w)
                for(int c=0; c<OutC; ++c) {
                    grad_flat[n*(H_out*W_out*OutC) + h*(W_out*OutC) + w*OutC + c] = grad_output.data[n*(OutC*H_out*W_out) + c*(H_out*W_out) + h*W_out + w];
                }
    Tensor grad_mat({N*H_out*W_out, OutC}, grad_flat); 
    
    Tensor dW_trans = matmul(transpose(col_matrix), grad_mat);
    Tensor dW = transpose(dW_trans); 
    
    return dW; 
}


Tensor maxpool2d_backward(const Tensor& grad_output, const Tensor& input_shape_ref, const Tensor& indices) {
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

void adam_step_wd(Tensor& param, const Tensor& grad, Tensor& m, Tensor& v, float lr, float beta1, float beta2, float eps, int t, float weight_decay) {
    if (param.size() != grad.size()) throw std::runtime_error("Adam size mismatch");
    float bc1 = 1.0f - std::pow(beta1, t);
    float bc2 = 1.0f - std::pow(beta2, t);
    for (size_t i = 0; i < param.size(); ++i) {
        // AdamW: decoupled weight decay applied directly to params
        float g = grad.data[i];
        m.data[i] = beta1 * m.data[i] + (1.0f - beta1) * g;
        v.data[i] = beta2 * v.data[i] + (1.0f - beta2) * g * g;
        float m_hat = m.data[i] / bc1;
        float v_hat = v.data[i] / bc2;
        param.data[i] -= lr * (m_hat / (std::sqrt(v_hat) + eps) + weight_decay * param.data[i]);
    }
}


Tensor global_avg_pool2d(const Tensor& input) {
    // input: [N, C, H, W] -> output: [N, C]
    if (input.shape.size() != 4) throw std::runtime_error("global_avg_pool2d requires 4D tensor [N,C,H,W]");
    
    int N = input.shape[0];
    int C = input.shape[1];
    int H = input.shape[2];
    int W = input.shape[3];
    int spatial = H * W;
    
    std::vector<float> out_data(N * C, 0.0f);
    
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            float sum = 0.0f;
            int base = n * (C * H * W) + c * (H * W);
            for (int i = 0; i < spatial; ++i) {
                sum += input.data[base + i];
            }
            out_data[n * C + c] = sum / spatial;
        }
    }
    
    return Tensor({N, C}, out_data);
}

Tensor global_avg_pool2d_backward(const Tensor& grad_output, int H, int W) {
    // grad_output: [N, C] -> grad_input: [N, C, H, W]
    int N = grad_output.shape[0];
    int C = grad_output.shape[1];
    int spatial = H * W;
    float scale = 1.0f / spatial;
    
    std::vector<float> grad_data(N * C * H * W);
    
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            float val = grad_output.data[n * C + c] * scale;
            int base = n * (C * H * W) + c * (H * W);
            for (int i = 0; i < spatial; ++i) {
                grad_data[base + i] = val;
            }
        }
    }
    
    return Tensor({N, C, H, W}, grad_data);
}


std::pair<Tensor, Tensor> dropout(const Tensor& input, float p, bool training) {
    if (!training || p <= 0.0f) {
        // No dropout during eval or if p=0
        return {Tensor(input.shape, input.data), ones(input.shape)};
    }
    
    size_t sz = input.size();
    std::vector<float> out_data(sz);
    std::vector<float> mask_data(sz);
    
    static thread_local std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    float scale = 1.0f / (1.0f - p); // Inverted dropout scaling
    
    for (size_t i = 0; i < sz; ++i) {
        if (dist(gen) >= p) {
            mask_data[i] = scale;
            out_data[i] = input.data[i] * scale;
        } else {
            mask_data[i] = 0.0f;
            out_data[i] = 0.0f;
        }
    }
    
    return {Tensor(input.shape, out_data), Tensor(input.shape, mask_data)};
}

Tensor dropout_backward(const Tensor& grad_output, const Tensor& mask) {
    size_t sz = grad_output.size();
    std::vector<float> grad_data(sz);
    for (size_t i = 0; i < sz; ++i) {
        grad_data[i] = grad_output.data[i] * mask.data[i];
    }
    return Tensor(grad_output.shape, grad_data);
}


Tensor random_horizontal_flip(const Tensor& input, float p) {
    // input: [N, C, H, W]
    if (input.shape.size() != 4) throw std::runtime_error("random_horizontal_flip requires 4D tensor [N,C,H,W]");
    
    int N = input.shape[0];
    int C = input.shape[1];
    int H = input.shape[2];
    int W = input.shape[3];
    
    std::vector<float> out_data = input.data; // Copy
    
    static thread_local std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    for (int n = 0; n < N; ++n) {
        if (dist(gen) < p) {
            // Flip this sample horizontally
            for (int c = 0; c < C; ++c) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W / 2; ++w) {
                        int idx1 = n * (C * H * W) + c * (H * W) + h * W + w;
                        int idx2 = n * (C * H * W) + c * (H * W) + h * W + (W - 1 - w);
                        std::swap(out_data[idx1], out_data[idx2]);
                    }
                }
            }
        }
    }
    
    return Tensor(input.shape, out_data);
}



Tensor load_image(const std::string& path) {
    // Read using OpenCV C++
    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Warning: Could not read image " << path << std::endl;
        return Tensor({0}, {}); 
    }

    // Convert to float [0, 1]
    cv::Mat float_img;
    img.convertTo(float_img, CV_32F, 1.0/255.0);

    // Get C, H, W
    int H = float_img.rows;
    int W = float_img.cols;
    int C = float_img.channels();
    
    std::vector<float> data;
    data.reserve(C * H * W);

    // HWC -> CHW
    if (C == 3) {
         std::vector<cv::Mat> channels(3);
         cv::split(float_img, channels);
         for (int i=0; i<3; ++i) {
             float* ptr = (float*)channels[i].data;
             data.insert(data.end(), ptr, ptr + H*W);
         }
    } else {
         if (float_img.isContinuous()) {
             float* ptr = (float*)float_img.data;
             data.assign(ptr, ptr + H*W);
         } else {
              for(int i=0; i<H; ++i) {
                  float* ptr = float_img.ptr<float>(i);
                  data.insert(data.end(), ptr, ptr + W);
              }
         }
    }
    return Tensor({C, H, W}, data);
}

Tensor load_image_batch(const std::vector<std::string>& paths, int C, int H, int W) {
    int N = paths.size();
    long long total_size = (long long)N * C * H * W; // Avoid overflow
    std::vector<float> batch_data;
    batch_data.reserve(total_size);
    
    for (const auto& path : paths) {
        cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
        
        // Handle Missing
        if (img.empty()) {
             // Zero padding
             batch_data.insert(batch_data.end(), C*H*W, 0.0f);
             continue;
        }

        // Handle Channel conversion
        if (C == 1 && img.channels() == 3) {
             cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
        } else if (C == 3 && img.channels() == 1) {
             cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
        }
        
        // Handle Resize
        if (img.rows != H || img.cols != W) {
             cv::resize(img, img, cv::Size(W, H));
        }
        
        img.convertTo(img, CV_32F, 1.0/255.0);
        
        // HWC -> CHW
        if (C == 3) {
             std::vector<cv::Mat> channels(3);
             cv::split(img, channels);
             for(int c=0; c<3; ++c) {
                  float* ptr = (float*)channels[c].data;
                  batch_data.insert(batch_data.end(), ptr, ptr + H*W);
             }
        } else {
             // Grayscale
             float* ptr = (float*)img.data;
             batch_data.insert(batch_data.end(), ptr, ptr + H*W);
        }
    }
    
    return Tensor({N, C, H, W}, batch_data);
}


std::vector<int> argmax(const Tensor& input, int axis) {
    if (axis != 1) throw std::runtime_error("argmax only matches axis=1 for now");
    
    int N = input.shape[0];
    int C = input.shape[1];
    
    std::vector<int> result(N);
    for (int n=0; n<N; ++n) {
        float max_val = -1e30f;
        int max_idx = 0;
        for (int c=0; c<C; ++c) {
             if (input.data[n*C + c] > max_val) {
                 max_val = input.data[n*C+c];
                 max_idx = c;
             }
        }
        result[n] = max_idx;
    }
    return result;
}

Tensor random_uniform(const std::vector<int>& shape, float min_val, float max_val) {
    size_t size = 1;
    for(int s : shape) size *= s;
    
    std::vector<float> data(size);
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(min_val, max_val);
    
    for(size_t i=0; i<size; ++i) data[i] = dist(gen);
    
    return Tensor(shape, data);
}

Tensor zeros(const std::vector<int>& shape) {
    size_t size = 1;
    for(int s : shape) size *= s;
    return Tensor(shape, std::vector<float>(size, 0.0f));
}

Tensor ones(const std::vector<int>& shape) {
    size_t size = 1;
    for(int s : shape) size *= s;
    return Tensor(shape, std::vector<float>(size, 1.0f));
}

Tensor random_crop_with_padding(const Tensor& input, int pad, float p) {
    // input: [N, C, H, W]
    // With probability p per image: zero-pad by `pad` on each side, then random-crop back to [N, C, H, W]
    // With probability (1-p): pass through unchanged
    if (input.shape.size() != 4) throw std::runtime_error("random_crop_with_padding requires 4D tensor [N,C,H,W]");
    
    int N = input.shape[0];
    int C = input.shape[1];
    int H = input.shape[2];
    int W = input.shape[3];
    int img_size = C * H * W;
    
    std::vector<float> out_data(N * C * H * W);
    
    static thread_local std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<int> dist_h(0, 2 * pad);
    std::uniform_int_distribution<int> dist_w(0, 2 * pad);
    std::uniform_real_distribution<float> coin(0.0f, 1.0f);
    
    for (int n = 0; n < N; ++n) {
        int base = n * img_size;
        
        if (coin(gen) > p) {
            // Pass through unchanged
            std::copy(input.data.begin() + base, 
                      input.data.begin() + base + img_size,
                      out_data.begin() + base);
            continue;
        }
        
        // Random crop offset for this sample
        int crop_y = dist_h(gen);
        int crop_x = dist_w(gen);
        
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    // Map to original image coords
                    int orig_h = crop_y + h - pad;
                    int orig_w = crop_x + w - pad;
                    
                    int dst_idx = base + c * (H * W) + h * W + w;
                    
                    if (orig_h >= 0 && orig_h < H && orig_w >= 0 && orig_w < W) {
                        int src_idx = base + c * (H * W) + orig_h * W + orig_w;
                        out_data[dst_idx] = input.data[src_idx];
                    } else {
                        out_data[dst_idx] = 0.0f;  // Zero padding
                    }
                }
            }
        }
    }
    
    return Tensor(input.shape, out_data);
}

} // namespace ops
