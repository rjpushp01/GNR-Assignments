#pragma once
#include <vector>
#include <iostream>
#include <numeric>

class Tensor {
public:
    std::vector<float> data;
    std::vector<int> shape;

    Tensor(const std::vector<int>& shape, const std::vector<float>& data = {});

    int size() const;
    void print() const;
    
    // Basic Ops (In-place/New Tensor logic to be decided, simpler to return new Tensors usually)
    Tensor add(const Tensor& other) const;
    Tensor mul(const Tensor& other) const;
};
