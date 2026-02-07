from .tensor import Tensor
try:
    from . import my_backend as mb
except ImportError:
    import my_backend as mb
import numpy as np

class Optimizer:
    def __init__(self, parameters, lr):
        self.parameters = tuple(parameters) # Make immutable
        self.lr = lr
        
    def zero_grad(self):
        for p in self.parameters:
            p.grad = None
            
    def step(self):
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, parameters, lr=0.01):
        super().__init__(parameters, lr)

    def step(self):
        for p in self.parameters:
            if p.grad is not None:
                if isinstance(p.grad, Tensor):
                    # Efficient C++ Update
                    # p.data is the C++ Tensor object
                    # p.grad.data is the C++ Tensor object (gradient)
                    mb.ops.sgd_step(p.data, p.grad.data, self.lr)
                else:
                    # Should not occur
                    continue

class Adam(Optimizer):
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(parameters, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        
        # State: m and v for each parameter
        self.m = []
        self.v = []
        
        for p in self.parameters:
            # Initialize moments as zeros matching parameter shape
            # Need to create zero tensor in C++.
            # Helper: numpy zeros -> Tensor.
            shape = p.shape # Tensor shape
            
            # Efficiently create zeros. Using p.data.shape from C++?
            # tensor.py wrapper exposes shape.
             
            # Construct zeros using list/numpy. 
            # In optim init, speed is less critical than per-step.
            size = np.prod(shape)
            zeros = mb.Tensor(shape, [0.0]*size)
            
            self.m.append(zeros)
            
            zeros_v = mb.Tensor(shape, [0.0]*size)
            self.v.append(zeros_v)
            
    def step(self):
        self.t += 1
        for i, p in enumerate(self.parameters):
            if p.grad is not None:
                # C++ Update
                mb.ops.adam_step(p.data, p.grad.data, self.m[i], self.v[i], 
                                 self.lr, self.beta1, self.beta2, self.eps, self.t)
