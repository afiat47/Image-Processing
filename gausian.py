import numpy as np
import kernel_normalize as k

def gaussian_kernel(sigmaX,sigmaY,sizeX=None,sizeY=None):

    if sizeX is None:
        sizeX = int(5*sigmaX)
    if sizeY is None:
        sizeY = int(5*sigmaY)

    if sizeY%2 == 0:
        sizeY -= 1
    if sizeX%2 == 0:
        sizeX -= 1

    centerX = (sizeX) // 2
    centerY = (sizeY) // 2
        
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigmaX * sigmaY)) * np.exp(-(((x-centerX)**2 / (sigmaX**2)) + ((y-centerY)**2 / (sigmaY**2))) / 2 )
         , (sizeX, sizeY)
        )
    
    kernel /= np.sum(kernel)

    newKernel = kernel.copy()

    print(kernel)

    newKernel = k.normalizedKernel(matrix=newKernel)
    return kernel

