import my_backend as mb

class Tensor:
    def __init__(self, data, shape=None, requires_grad=False, _ctx=None):
        self.grad = None
        self.requires_grad = requires_grad
        self._ctx = _ctx # Function that created this tensor (for backward)
        
        if isinstance(data, mb.Tensor):
            self.data = data
        else:
            if isinstance(data, list):
                 # Flatten list if needed?
                 # Assuming data is flat list or scalar for now
                 if not data:
                      self.data = mb.Tensor([0], [])
                 else:
                      # Check if list of numbers
                      if shape is None:
                           # Simple 1D inference
                           shape = [len(data)]
                      
                      # Convert to flat float list
                      # If nested, we need a recursive flatten. 
                      # For assignment, let's assume flat data passed or use helper
                      flat_data = self._flatten(data)
                      self.data = mb.Tensor(shape, flat_data)
            elif isinstance(data, (float, int)):
                 self.data = mb.Tensor([1], [float(data)])
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")
                
        self.shape = self.data.shape

    def _flatten(self, l):
        if not isinstance(l, list):
            return [float(l)]
        out = []
        for x in l:
            out.extend(self._flatten(x))
        return out

    def __repr__(self):
        return f"Tensor({self.data.__repr__()}, requires_grad={self.requires_grad})"

    def to_list(self):
        # Convert C++ binding data (which exposes .data vector) to list
        return self.data.data

    def item(self):
        # For scalar tensors
        return self.data.data[0]

    def print(self):
        self.data.print()

    # --- Autograd Engine ---
    def backward(self, grad=None):
        if self._ctx is None:
            return
        
        if grad is None:
            # Scalar output, default to ones
            grad = Tensor(mb.ops.ones(self.shape), requires_grad=False)
        
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
                # print(f"Node: {type(node._ctx).__name__}, Parents: {len(node._ctx.parents)}, Grads type: {type(grads)}")
                if len(node._ctx.parents) == 1:
                    if not isinstance(grads, (list, tuple)):
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
        
    def reshape(self, shape):
        # shape: list or tuple of ints
        out_data = self.data.reshape(list(shape)) # returns new C++ Tensor
        # Gradients? Reshape is differentiable (view). 
        # C++ reshape returns new tensor sharing data (copy for now).
        out = Tensor(out_data, requires_grad=self.requires_grad)
        if out.requires_grad:
             out._ctx = ReshapeBackward(self, self.shape)
        return out

    def dropout(self, p=0.5, training=True):
        # C++ returns (output, mask)
        res = mb.ops.dropout(self.data, p, training)
        out = Tensor(res[0], requires_grad=self.requires_grad)
        if out.requires_grad and training:
             out._ctx = DropoutBackward(self, res[1])
        return out
        
# --- Backward Functions (Autograd Nodes) ---

class Function:
    def __init__(self, *parents):
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
        grad_input = Tensor(mb.ops.conv2d_backward_input(grad_output.data, self.kernel.data, self.stride, self.padding, self.input.data), requires_grad=False)
        grad_kernel = Tensor(mb.ops.conv2d_backward_kernel(grad_output.data, self.input.data, self.col_matrix, self.stride, self.padding), requires_grad=False)
        return grad_input, grad_kernel
        
class ReshapeBackward(Function):
     def __init__(self, x, old_shape):
         super().__init__(x)
         self.x = x
         self.old_shape = old_shape
     def backward(self, grad_output):
         # Reshape grad back to old shape
         grad = grad_output.reshape(self.old_shape)
         return grad
        
class MaxPool2dBackward(Function):
    def __init__(self, input, indices):
        super().__init__(input)
        self.input = input
        self.indices = indices
        
    def backward(self, grad_output):
        grad_input = Tensor(mb.ops.maxpool2d_backward(grad_output.data, self.input.data, self.indices), requires_grad=False)
        return grad_input

class DropoutBackward(Function):
    def __init__(self, input, mask):
        super().__init__(input)
        self.mask = mask  
    
    def backward(self, grad_output):
        grad = Tensor(mb.ops.dropout_backward(grad_output.data, self.mask), requires_grad=False)
        return grad
