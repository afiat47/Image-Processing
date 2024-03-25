import numpy as np
import cv2

def split_region(image,min_size=2):

    if image.shape[0] <= min_size and image.shape[1] <= min_size:
        image[:, :, 0].fill(np.mean(image[:, :, 0]))
        image[:, :, 1].fill(np.mean(image[:, :, 1]))
        image[:, :, 2].fill(np.mean(image[:, :, 2]))
        return image

    if np.std(image[:, :, 0]) < threshold and np.std(image[:, :, 1]) < threshold and np.std(image[:, :, 2]) < threshold:

        image[:, :, 0].fill(np.mean(image[:, :, 0]))
        image[:, :, 1].fill(np.mean(image[:, :, 1]))
        image[:, :, 2].fill(np.mean(image[:, :, 2]))
        return image

    Mx = image.shape[0] // 2
    My = image.shape[1] // 2

    regions=[
    image[: Mx, : My],
    image[:Mx, My:],
    image[Mx:,: My],
    image[Mx:,My:]
    ]

    for region in regions:
        split_region(region)

    return

img = cv2.imread('Lena.jpg', cv2.IMREAD_COLOR)
cv2.imshow("RGB", img)
cv2.waitKey(0)
threshold = int(input(" Threshold: "))
split_region(img)
cv2.imshow("Segmented Image", img)
cv2.waitKey(0)

cv2.destroyAllWindows()