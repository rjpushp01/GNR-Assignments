
import sys
sys.path.append('python')
import my_framework as mf
import numpy as np

def test_matmul_backward():
    print("Testing MatMul Backward...")
    a = mf.Tensor([[1, 2], [3, 4]], requires_grad=True)
    b = mf.Tensor([[1, 0], [0, 1]], requires_grad=True)
    c = a.matmul(b)
    d = mf.Tensor([[1, 1], [1, 1]], requires_grad=False)
    
    c.backward(d)
    
    # dL/dA = dL/dC * B^T = [[1,1], [1,1]] * [[1,0], [0,1]]^T = [[1,1], [1,1]]
    expected_grad_a = np.ones((2, 2)) @ b.numpy().T
    
    # dL/dB = A^T * dL/dC = [[1,2], [3,4]]^T * [[1,1], [1,1]] = [[1,3], [2,4]] * [[1,1], [1,1]] = [[4,4], [6,6]]
    expected_grad_b = a.numpy().T @ np.ones((2, 2))
    
    print("Grad A:\n", a.grad.numpy())
    print("Expected A:\n", expected_grad_a)
    assert np.allclose(a.grad.numpy(), expected_grad_a)
    
    print("Grad B:\n", b.grad.numpy())
    print("Expected B:\n", expected_grad_b)
    assert np.allclose(b.grad.numpy(), expected_grad_b)
    
    print("MatMul Backward Passed!")

def test_relu_backward():
    print("Testing ReLU Backward...")
    a = mf.Tensor([[-1, 2], [3, -4]], requires_grad=True)
    b = a.relu()
    grad = mf.Tensor([[1, 1], [1, 1]], requires_grad=False)
    b.backward(grad)
    
    # Expected: [[0, 1], [1, 0]]
    print("Grad A (ReLU):\n", a.grad.numpy())
    assert np.allclose(a.grad.numpy(), [[0, 1], [1, 0]])
    print("ReLU Backward Passed!")

if __name__ == "__main__":
    try:
        test_matmul_backward()
        test_relu_backward()
        print("All Backward Tests Passed!")
    except Exception as e:
        print("Test Failed:", e)
        import traceback
        traceback.print_exc()
