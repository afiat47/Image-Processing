import numpy as np
import kernel_normalize as k

def mean(height=5, width=6):
    kernel = np.ones((height, width), dtype=np.float32)
    kernel = kernel / (height * width)

    print(kernel)
    newKernel = kernel.copy()
    newKernel = k.normalizedKernel(matrix=newKernel)
    return kernel