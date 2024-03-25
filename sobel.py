import kernel_normalize as k
def sobel(kernel1, kernel2):
    kernel = kernel1 * kernel2
    print(kernel)
    # newKernel = kernel.copy()
    # newKernel = k.normalizedKernel(matrix=newKernel)
    return kernel

def sobel2(kernel1, kernel2):
    kernel = kernel2 * kernel1
    print(kernel)
    # newKernel = kernel.copy()
    # newKernel = k.normalizedKernel(matrix=newKernel)
    return kernel