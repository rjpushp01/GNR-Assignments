from .tensor import Tensor
try:
    from . import my_backend as mb
except ImportError:
    try:
        import my_framework.my_backend as mb
    except ImportError:
        import my_backend as mb

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
                    mb.ops.sgd_step(p.data, p.grad.data, self.lr)

class Adam(Optimizer):
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        super().__init__(parameters, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        
        # State: m and v for each parameter
        self.m = []
        self.v = []
        
        for p in self.parameters:
            # Initialize moments as zeros matching parameter shape
            # Using C++ zeros op
            
            # p.shape is a list. C++ expects std::vector<int>
            zeros = mb.ops.zeros(p.shape)
            
            # Wrap in Tensor (which wraps the underlying C++ Tensor)
            # Wait, mb.ops.zeros returns a C++ Tensor object.
            # Our python Tensor class expects `data` to be C++ Tensor or list.
            # So `Tensor(zeros)` works.
            
            self.m.append(Tensor(zeros))
            self.v.append(Tensor(mb.ops.zeros(p.shape)))
            
    def step(self):
        self.t += 1
        for i, p in enumerate(self.parameters):
            if p.grad is not None:
                if self.weight_decay > 0:
                    # AdamW: decoupled weight decay
                    mb.ops.adam_step_wd(p.data, p.grad.data, self.m[i].data, self.v[i].data, 
                                       self.lr, self.beta1, self.beta2, self.eps, self.t, self.weight_decay)
                else:
                    # Standard Adam
                    mb.ops.adam_step(p.data, p.grad.data, self.m[i].data, self.v[i].data, 
                                    self.lr, self.beta1, self.beta2, self.eps, self.t)
