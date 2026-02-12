#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "tensor.hpp"
#include "ops.hpp"

namespace py = pybind11;

PYBIND11_MODULE(my_backend, m) {
    m.doc() = "GNR 638 Assignment 1 C++ Backend";

    py::class_<Tensor>(m, "Tensor")
        .def(py::init<const std::vector<int>&, const std::vector<float>&>(),
             py::arg("shape"), py::arg("data") = std::vector<float>())
        .def("size", &Tensor::size)
        .def("print", &Tensor::print)
        .def("add", &Tensor::add)
        .def("mul", &Tensor::mul)
        .def("reshape", &Tensor::reshape)
        .def_readwrite("data", &Tensor::data)
        .def_readwrite("shape", &Tensor::shape)
        .def("__repr__", [](const Tensor &t) {
             std::string shape_str = "[";
             for (size_t i = 0; i < t.shape.size(); ++i) {
                 shape_str += std::to_string(t.shape[i]) + (i < t.shape.size() - 1 ? ", " : "");
             }
             shape_str += "]";
             return "<my_backend.Tensor shape=" + shape_str + ">";
        });

    // Submodule for ops
    auto m_ops = m.def_submodule("ops", "Operations");
    
    // Forward Ops returning multiple tensors (Conv, Pool)
    m_ops.def("conv2d", &ops::conv2d);
    m_ops.def("maxpool2d", &ops::maxpool2d);
    
    // Basic Ops
    m_ops.def("matmul", &ops::matmul);
    m_ops.def("transpose", &ops::transpose);
    m_ops.def("relu", &ops::relu);
    
    // Backward Ops
    m_ops.def("matmul_backward_input", &ops::matmul_backward_input);
    m_ops.def("matmul_backward_other", &ops::matmul_backward_other);
    m_ops.def("relu_backward", &ops::relu_backward);
    
    m_ops.def("conv2d_backward_input", &ops::conv2d_backward_input);
    m_ops.def("conv2d_backward_kernel", &ops::conv2d_backward_kernel);
    
    m_ops.def("maxpool2d_backward", &ops::maxpool2d_backward);
    
    // Loss
    m_ops.def("cross_entropy", &ops::cross_entropy);
    m_ops.def("cross_entropy_backward", &ops::cross_entropy_backward);
    
    // Optimizers
    m_ops.def("sgd_step", &ops::sgd_step);
    m_ops.def("adam_step", &ops::adam_step);
    m_ops.def("adam_step_wd", &ops::adam_step_wd);
    
    // Dropout
    m_ops.def("dropout", &ops::dropout);
    m_ops.def("dropout_backward", &ops::dropout_backward);
    
    // Data Augmentation
    m_ops.def("random_horizontal_flip", &ops::random_horizontal_flip);
    m_ops.def("random_crop_with_padding", &ops::random_crop_with_padding);
    
    // New Ops
    m_ops.def("load_image", &ops::load_image);
    m_ops.def("load_image_batch", &ops::load_image_batch);
    m_ops.def("argmax", &ops::argmax);
    m_ops.def("random_uniform", &ops::random_uniform);
    m_ops.def("zeros", &ops::zeros);
    m_ops.def("ones", &ops::ones);
}
