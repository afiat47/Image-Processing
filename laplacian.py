import numpy as np
import kernel_normalize as k

def laplacian_mask(n,m):
    kernel = - np.ones((n, m))
    centerx = n // 2
    centery = m // 2
    kernel[centerx,centery] = - (1 - n * m)

    print(kernel)
    # newKernel = kernel.copy()
    # newKernel = k.normalizedKernel(matrix=newKernel)
    return kernel
