#include "tensor.hpp"
#include <stdexcept>
#include <random>

Tensor::Tensor(const std::vector<int>& shape, const std::vector<float>& data) : shape(shape) {
    int total_size = 1;
    for (int s : shape) total_size *= s;
    
    if (data.empty()) {
        this->data.resize(total_size, 0.0f);
    } else {
        if (data.size() != total_size) {
            throw std::runtime_error("Data size does not match shape: Expected " + std::to_string(total_size) + " but got " + std::to_string(data.size()));
        }
        this->data = data;
    }
}

int Tensor::size() const {
    return data.size();
}

void Tensor::print() const {
    std::cout << "Tensor(";
    if (shape.size() > 0) {
        std::cout << "shape=[";
        for (size_t i = 0; i < shape.size(); ++i) {
            std::cout << shape[i] << (i < shape.size() - 1 ? ", " : "");
        }
        std::cout << "], data_snippet=[";
        int limit = std::min((int)data.size(), 5);
        for (int i = 0; i < limit; ++i) {
            std::cout << data[i] << (i < limit - 1 ? ", " : "");
        }
        if (data.size() > 5) std::cout << "...";
        std::cout << "]";
    }
    std::cout << ")" << std::endl;
}

Tensor Tensor::add(const Tensor& other) const {
    if (size() != other.size()) throw std::runtime_error("Shape mismatch in add (broadcasting not fully supported yet)");
    
    std::vector<float> result_data(size());
    for (size_t i = 0; i < size(); ++i) {
        result_data[i] = data[i] + other.data[i];
    }
    return Tensor(shape, result_data);
}

Tensor Tensor::mul(const Tensor& other) const {
     if (size() != other.size()) throw std::runtime_error("Shape mismatch in mul");

    std::vector<float> result_data(size());
    for (size_t i = 0; i < size(); ++i) {
        result_data[i] = data[i] * other.data[i];
    }
    return Tensor(shape, result_data);
}
