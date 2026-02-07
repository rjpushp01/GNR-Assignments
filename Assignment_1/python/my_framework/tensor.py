import numpy as np
import my_backend as mb

class Tensor:
    def __init__(self, data, shape=None, requires_grad=False, _ctx=None):
        self.grad = None
        self.requires_grad = requires_grad
        self._ctx = _ctx # Function that created this tensor (for backward)
        
        if isinstance(data, mb.Tensor):
            self.data = data
        else:
            if isinstance(data, (np.ndarray, list)):
                if isinstance(data, list):
                    data = np.array(data, dtype=np.float32)
                else:
                    data = data.astype(np.float32)
                
                if shape is None:
                    shape = list(data.shape)
                self.data = mb.Tensor(shape, data.flatten().tolist())
            elif isinstance(data, (float, int)):
                 self.data = mb.Tensor([1], [float(data)])
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")
                
        self.shape = self.data.shape

    def __repr__(self):
        # Using C++ print via capture might be hard, so just reconstruct numpy or use simple repr
        return f"Tensor({self.data.__repr__()}, requires_grad={self.requires_grad})"

    def numpy(self):
        # Copy data back to numpy (not efficient but okay for this assignment)
        return np.array(self.data.data).reshape(self.shape)

    def print(self):
        self.data.print()

    # --- Autograd Engine ---
    def backward(self, grad=None):
        if self._ctx is None:
            return
        
        if grad is None:
            # Assume scalar output
            grad = Tensor(np.ones(self.shape), requires_grad=False)
        
        self.grad = grad
        
        # Topological Sort
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                if v._ctx:
                    for parent in v._ctx.parents:
                        build_topo(parent)
                topo.append(v)
        
        build_topo(self)
        
        # Backward Pass
        for node in reversed(topo):
            if node._ctx:
                grads = node._ctx.backward(node.grad)
                if len(node._ctx.parents) == 1:
                    grads = [grads]
                
                for parent, parent_grad in zip(node._ctx.parents, grads):
                    if parent.requires_grad:
                        if parent.grad is None:
                            parent.grad = parent_grad
                        else:
                            # Accumulate gradient
                            parent.grad = parent.grad + parent_grad # Using __add__

    # --- Operations ---
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data.add(other.data), requires_grad=(self.requires_grad or other.requires_grad))
        if out.requires_grad:
            out._ctx = AddBackward(self, other)
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data.mul(other.data), requires_grad=(self.requires_grad or other.requires_grad))
        if out.requires_grad:
            out._ctx = MulBackward(self, other)
        return out
        
    def matmul(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(mb.ops.matmul(self.data, other.data), requires_grad=(self.requires_grad or other.requires_grad))
        if out.requires_grad:
            out._ctx = MatMulBackward(self, other)
        return out
        
    def relu(self):
        out = Tensor(mb.ops.relu(self.data), requires_grad=self.requires_grad)
        if out.requires_grad:
            out._ctx = ReluBackward(self)
        return out

    def conv2d(self, kernel, stride=1, padding=0):
        # kernel is a Tensor
        # C++ returns (output, col_matrix)
        res = mb.ops.conv2d(self.data, kernel.data, stride, padding)
        out = Tensor(res[0], requires_grad=(self.requires_grad or kernel.requires_grad))
        if out.requires_grad:
             # Store col_matrix (res[1]) in context for backward
             out._ctx = Conv2dBackward(self, kernel, res[1], stride, padding)
        return out

    def maxpool2d(self, kernel_size=2, stride=2):
        # C++ returns (output, indices)
        res = mb.ops.maxpool2d(self.data, kernel_size, stride)
        out = Tensor(res[0], requires_grad=self.requires_grad)
        if out.requires_grad:
             out._ctx = MaxPool2dBackward(self, res[1])
        return out
        
# --- Backward Functions (Autograd Nodes) ---

class Function:
    def __init__(self, *parents):
        # Only store parents that require grad to save memory? 
        # No, simpler to store all to maintain graph structure logic for now.
        self.parents = parents
    
    def backward(self, grad_output):
        raise NotImplementedError

class AddBackward(Function):
    def backward(self, grad_output):
        return grad_output, grad_output

class MulBackward(Function):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.x = x
        self.y = y
        
    def backward(self, grad_output):
        # Element-wise mul gradients
        # dA = dC * B
        # dB = dC * A
        grad_x = Tensor(grad_output.data.mul(self.y.data), requires_grad=False)
        grad_y = Tensor(grad_output.data.mul(self.x.data), requires_grad=False)
        return grad_x, grad_y

class MatMulBackward(Function):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.x = x
        self.y = y
        
    def backward(self, grad_output):
        grad_x = Tensor(mb.ops.matmul_backward_input(grad_output.data, self.y.data), requires_grad=False)
        grad_y = Tensor(mb.ops.matmul_backward_other(grad_output.data, self.x.data), requires_grad=False)
        return grad_x, grad_y

class ReluBackward(Function):
    def __init__(self, x):
         super().__init__(x)
         self.x = x

    def backward(self, grad_output):
        grad = Tensor(mb.ops.relu_backward(grad_output.data, self.x.data), requires_grad=False)
        return grad

class Conv2dBackward(Function):
    def __init__(self, input, kernel, col_matrix, stride, padding):
        super().__init__(input, kernel)
        self.input = input
        self.kernel = kernel
        self.col_matrix = col_matrix
        self.stride = stride
        self.padding = padding
        
    def backward(self, grad_output):
        # Optimized backward using stored col_matrix
        grad_input = Tensor(mb.ops.conv2d_backward_input(grad_output.data, self.kernel.data, self.stride, self.padding, self.input.data), requires_grad=False)
        grad_kernel = Tensor(mb.ops.conv2d_backward_kernel(grad_output.data, self.input.data, self.col_matrix, self.stride, self.padding), requires_grad=False)
        return grad_input, grad_kernel
        
class MaxPool2dBackward(Function):
    def __init__(self, input, indices):
        super().__init__(input)
        self.input = input
        self.indices = indices
        
    def backward(self, grad_output):
        # Uses indices for efficient backward
        grad_input = Tensor(mb.ops.maxpool2d_backward(grad_output.data, self.input.data, self.indices), requires_grad=False)
        return grad_input


