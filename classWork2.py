import numpy as np
import cv2

LinaImage = "D:\\project\\pythonProject\\Lena.jpg"
def convolution(image, kernel, centerx=None, centery=None):
    image_height = image.shape[0]
    image_width = image.shape[1]

    kernel_height = kernel.shape[0]
    kernel_width = kernel.shape[1]

    if centerx is None:
        centerx = kernel_height // 2
    if centery is None:
        centery = kernel_width // 2

    if centerx > kernel_height or centery > kernel_width or centerx < 0 or centery < 0:
        print("Invalid center of the kernel")
        return None

    image = cv2.copyMakeBorder(src=image, top=centerx, bottom=kernel_height - centerx - 1, left=centery,
                               right=kernel_width - centery - 1, borderType=cv2.BORDER_REFLECT)

    # # Updating image height and width
    image_height = image.shape[0]
    image_width = image.shape[1]

    output_image = np.zeros((image_height, image_width))

    for i in range(centerx, image_height - centerx):
        for j in range(centery, image_width - centery):
            sum = 0
            for k in range(kernel_height):
                for l in range(kernel_width):
                    sum += kernel[kernel_height - k - 1][kernel_width - l - 1] * image[i - centerx + k][j - centery + l]
            output_image[i][j] = sum

    return (output_image)


def laplacian_of_gaussian(size, sigma):

    kernel = np.zeros((size, size))
    for x in range(size):
        for y in range(size):
            kernel[x, y] = ((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2 - 2 * sigma ** 2) * np.exp(
                -((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2))
    return kernel

laplacian = laplacian_of_gaussian(7, 1)
print("\nLaplacian of Gaussian (LoG) Kernel:")
print(laplacian)

img = cv2.imread(LinaImage)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
out=convolution(img,laplacian)

lapimage=out.copy()

cv2.normalize(out, out,0, 255, cv2.NORM_MINMAX)
out = np.round(out).astype(np.uint8)
cv2.imshow("Output", out)
cv2.waitKey(0)
def find_edges(matrix):
    rows, cols = matrix.shape
    edge_img = np.zeros_like(matrix)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if (matrix[i-1, j] * matrix[i + 1, j] < 0) or \
                    (matrix[i, j+1] * matrix[i, j-1] < 0):
                pad = matrix.shape[0] // 2
                local_region = matrix[i - pad:i + pad + 1, j - pad:j + pad + 1]
                local_stddev = np.std(local_region)
                if local_stddev*local_stddev>60:
                    edge_img[i, j] = 255
                    edge_img= np.round(edge_img).astype(np.uint8)


    return edge_img

zero_crossings = find_edges(lapimage)
cv2.imshow("Edge Image",lapimage)

cv2.waitKey(0)
print("\nZero Crossings:")
print(zero_crossings)