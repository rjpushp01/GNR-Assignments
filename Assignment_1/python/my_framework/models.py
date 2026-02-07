from .tensor import Tensor, Function
try:
    from . import my_backend as mb
except ImportError:
    try:
        import my_framework.my_backend as mb
    except ImportError:
        import my_backend as mb

import numpy as np

class Module:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError
        
    def parameters(self):
        params = []
        for name, value in self.__dict__.items():
            if isinstance(value, Tensor):
                if value.requires_grad:
                    params.append(value)
            elif isinstance(value, Module):
                params.extend(value.parameters())
            elif isinstance(value, list):
                for item in value:
                     if isinstance(item, Module):
                         params.extend(item.parameters())
        return params

    def train(self):
        self.training = True
        for name, value in self.__dict__.items():
            if isinstance(value, Module):
                value.train()
                
    def eval(self):
        self.training = False
        for name, value in self.__dict__.items():
            if isinstance(value, Module):
                value.eval()

class Linear(Module):
    def __init__(self, in_features, out_features):
        # Xavier Initialization
        limit = np.sqrt(6 / (in_features + out_features))
        self.weight = Tensor(np.random.uniform(-limit, limit, (in_features, out_features)), requires_grad=True)
        self.bias = Tensor(np.zeros(out_features), requires_grad=True) 
        self.out_features = out_features

    def forward(self, x):
        out = x.matmul(self.weight)
        return out

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # He Initialization
        limit = np.sqrt(2 / (in_channels * kernel_size * kernel_size))
        self.weight = Tensor(np.random.uniform(-limit, limit, (out_channels, in_channels, kernel_size, kernel_size)), requires_grad=True)
        
    def forward(self, x):
        return x.conv2d(self.weight, self.stride, self.padding)

class ReLU(Module):
    def forward(self, x):
        return x.relu()

class MaxPool2d(Module):
    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride
        
    def forward(self, x):
        return x.maxpool2d(self.kernel_size, self.stride)

class ReshapeBackward(Function):
     def __init__(self, x, old_shape):
         super().__init__(x)
         self.x = x
         self.old_shape = old_shape
     def backward(self, grad_output):
         return Tensor(grad_output.data.data, shape=self.old_shape, requires_grad=False)

class Flatten(Module):
    def forward(self, x):
        N = x.shape[0]
        size = 1
        for s in x.shape[1:]: size *= s
        
        new_shape = [N, size]
        
        out = Tensor(x.data.data, shape=new_shape, requires_grad=x.requires_grad)
        if x.requires_grad:
             out._ctx = ReshapeBackward(x, x.shape)
        return out

class CrossEntropyBackward(Function):
    def __init__(self, input, target):
        super().__init__(input, target) 
        self.input = input
        self.target = target
        
    def backward(self, grad_output):
        grad = Tensor(mb.ops.cross_entropy_backward(self.input.data, self.target.data), requires_grad=False)
        return grad, None 

class CrossEntropyLoss(Module):
    def forward(self, input, target):
        loss_val = Tensor(mb.ops.cross_entropy(input.data, target.data), requires_grad=True)
        loss_val._ctx = CrossEntropyBackward(input, target)
        return loss_val

# Models
class MNIST_Model(Module):
    def __init__(self):
        # LeNet-ish
        # Input: 1x28x28
        self.conv1 = Conv2d(1, 6, 5) # -> 6x24x24
        self.relu1 = ReLU()
        self.pool1 = MaxPool2d(2, 2) # -> 6x12x12
        self.conv2 = Conv2d(6, 16, 5) # -> 16x8x8
        self.relu2 = ReLU()
        self.pool2 = MaxPool2d(2, 2) # -> 16x4x4
        self.flatten = Flatten()
        self.fc1 = Linear(16*4*4, 120)
        self.relu3 = ReLU()
        self.fc2 = Linear(120, 84)
        self.relu4 = ReLU()
        self.fc3 = Linear(84, 10)
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x

class CIFAR_Model(Module):
    def __init__(self):
        # VGG-Style (Deeper) for CIFAR-100
        # Input: 3x32x32
        
        # Block 1
        self.conv1 = Conv2d(3, 32, 3, padding=1)  # -> 32x32x32
        self.relu1 = ReLU()
        self.conv2 = Conv2d(32, 64, 3, padding=1) # -> 64x32x32
        self.relu2 = ReLU()
        self.pool1 = MaxPool2d(2, 2)              # -> 64x16x16
        
        # Block 2
        self.conv3 = Conv2d(64, 128, 3, padding=1)# -> 128x16x16
        self.relu3 = ReLU()
        self.conv4 = Conv2d(128, 128, 3, padding=1) # -> 128x16x16
        self.relu4 = ReLU()
        self.pool2 = MaxPool2d(2, 2)               # -> 128x8x8
        
        # Block 3
        self.conv5 = Conv2d(128, 256, 3, padding=1)# -> 256x8x8
        self.relu5 = ReLU()
        self.pool3 = MaxPool2d(2, 2)               # -> 256x4x4
        
        self.flatten = Flatten()
        
        # Classifier
        self.fc1 = Linear(256*4*4, 1024)
        self.relu6 = ReLU()
        self.fc2 = Linear(1024, 512)
        self.relu7 = ReLU()
        self.fc3 = Linear(512, 100) # CIFAR-100
        
    def forward(self, x):
        # Block 1
        x = self.pool1(self.relu2(self.conv2(self.relu1(self.conv1(x)))))
        
        # Block 2
        x = self.pool2(self.relu4(self.conv4(self.relu3(self.conv3(x)))))
        
        # Block 3
        x = self.pool3(self.relu5(self.conv5(x)))
        
        x = self.flatten(x)
        x = self.relu6(self.fc1(x))
        x = self.relu7(self.fc2(x))
        x = self.fc3(x)
        return x
