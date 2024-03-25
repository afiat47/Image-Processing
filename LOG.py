import cv2
import kernel_normalize as k
from scipy.signal import convolve2d
def LOG(kernel1,kernel2):
    kernel = convolve2d(kernel1, kernel2, mode='same')
    print(kernel)
    newKernel = kernel.copy()
    newKernel = k.normalizedKernel(matrix=newKernel)
    return(kernel)