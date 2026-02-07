
import sys
import os
sys.path.append('python')
import my_framework as mf
import numpy as np

def test_matmul():
    print("Testing MatMul...")
    a = mf.Tensor([[1, 2], [3, 4]])
    b = mf.Tensor([[1, 0], [0, 1]])
    c = a.matmul(b)
    print("Result:\n", c.numpy())
    assert np.allclose(c.numpy(), [[1, 2], [3, 4]])
    print("MatMul Passed!")

def test_conv2d():
    print("Testing Conv2d...")
    # Input: 1 image, 1 channel, 4x4
    img = np.array([[[[1, 2, 3, 4], 
                      [5, 6, 7, 8], 
                      [9, 10, 11, 12], 
                      [13, 14, 15, 16]]]], dtype=np.float32)
    t_img = mf.Tensor(img)
    
    # Kernel: 1 filter, 1 channel, 2x2
    kernel = np.array([[[[1, 0], 
                         [0, 1]]]], dtype=np.float32) # Identity-ish
    t_kern = mf.Tensor(kernel)
    
    # Stride 1, padding 0 -> Output 3x3
    res = t_img.conv2d(t_kern, stride=1, padding=0)
    
    print("Conv Result Shape:", res.shape)
    print("Conv Result:\n", res.numpy())
    
    # Expected[0,0] = 1*1 + 2*0 + 5*0 + 6*1 = 7
    assert res.numpy()[0, 0, 0, 0] == 7.0
    print("Conv2d Passed!")

if __name__ == "__main__":
    try:
        test_matmul()
        test_conv2d()
        print("All Forward Tests Passed!")
    except Exception as e:
        print("Test Failed:", e)
        import traceback
        traceback.print_exc()

