import numpy as np
import cv2


# # def convolution(image,kernel,centerx=None,centery=None):

# #     kernel_height = kernel.shape[0]
# #     kernel_width = kernel.shape[1]

# #     if centerx is None:
# #         centerx = kernel_height // 2
# #     if centery is None:
# #         centery = kernel_width // 2

# #     kernel_midX = kernel_height // 2
# #     kernel_midY = kernel_width // 2   


# #     padding_bottom = kernel_height  - centerx - 1
# #     padding_right = kernel_width - centery  - 1

# #     img_bordered = cv2.copyMakeBorder(src=image, top=centerx, bottom=padding_bottom, left=centery, right=padding_right,
# #                                       borderType=cv2.BORDER_CONSTANT)
# #     output_image = np.zeros((img_bordered.shape[0], img_bordered.shape[1]))

# #     for i in range(centerx, img_bordered.shape[0] - padding_bottom - kernel_midX):
# #         for j in range(centery, img_bordered.shape[1] - padding_right - kernel_midY):
# #             sum = 0
# #             for x in range(-kernel_midX, kernel_midX + 1):
# #                 for y in range(-kernel_midY, kernel_midY + 1):
# #                     sum += kernel[x + kernel_midX, y + kernel_midY] * img_bordered[i - x, j - y]
# #             output_image[i, j] = sum

# #     return output_image



# # def convolution(image,kernel,centerx=None,centery=None):
# #     image_height = image.shape[0]
# #     image_width = image.shape[1]

# #     kernel_height = kernel.shape[0]
# #     kernel_width = kernel.shape[1]

# #     if centerx is None:
# #         centerx = kernel_height // 2
# #     if centery is None:
# #         centery = kernel_width // 2



# #     output_image = np.zeros((image_height, image_width))

# #     for i in range(centerx,image_height-centerx):
# #         for j in range(centery,image_width-centery):
# #             sum = 0
# #             for k in range(kernel_height):
# #                 for l in range(kernel_width):
# #                     sum += kernel[kernel_height-k-1][kernel_width-l-1] * image[i-centerx+k][j-centery+l] 
# #             output_image[i][j] = sum

# #     return(output_image)

# # def convolution(image, kernel, centerx=2, centery=2):
# #     k = kernel.shape[0] // 2
# #     l = kernel.shape[1] // 2

# #     padding_bottom = kernel.shape[0] - 1 - centerx
# #     padding_right = kernel.shape[1] - 1 - centery

# #     img_bordered = cv2.copyMakeBorder(src=image, top=centerx, bottom=padding_bottom, left=centery, right=padding_right,
# #                                       borderType=cv2.BORDER_CONSTANT)
# #     out = img_bordered.copy()

# #     for i in range(centerx, img_bordered.shape[0] - padding_bottom - k):
# #         for j in range(centery, img_bordered.shape[1] - padding_right - l):
# #             res = 0
# #             for x in range(-k, k + 1):
# #                 for y in range(-l, l + 1):
# #                     res += kernel[x + k, y + l] * img_bordered[i - x, j - y]
# #             out[i, j] = res

# #     return out




# def pad_image(image, kernel_height, kernel_width, kernel_center):
#     pad_top = kernel_center[0]
#     pad_bottom = kernel_height - kernel_center[0] - 1
#     pad_left = kernel_center[1]
#     pad_right = kernel_width - kernel_center[1] - 1
    
#     padded_image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values = 0)
#     return padded_image

# def convolution(image, kernel, centerx,centery):    
#     kernel_height, kernel_width = len(kernel), len(kernel[0])
    
#     kernel_center = (centery, centerx)
    
#     # pad the input image based on kernel and center
#     padded_image = pad_image(image = image,  kernel_height = kernel_height, kernel_width = kernel_width, kernel_center = kernel_center)

#     # generating output with dummy zeros(0)
#     output = np.zeros_like(padded_image, dtype='float32')
    
#     #print("Padded image")
#     #print(padded_image)
    
#     # xx = 1
#     # yy = 2
#     # print(f"Value at ({xx},{yy}) is {padded_image[xx,yy]}")
    
#     # padded image height, width
#     padded_height, padded_width = padded_image.shape

#     kcx = kernel_center[0]
#     kcy = kernel_center[1]
    
#     # iterating through height. For (1,1) kernel, it iterates from 1 to (h - 1)
#     for x in range( kcx, padded_height - ( kernel_height - (kcx+1)) ):
#         # iterate through width. For (1,1) kernel, it iterates from 1 to (w - 1)
#         for y in range( kcy, padded_width - ( kernel_width - (kcy + 1)) ):
            
#             # calculating the portion in image, that will be convoluted now
#             image_start_x = x - kcx
#             image_start_y = y - kcy
            
#             # if x == 1 and y == 2:
#             #     print(f"For position({x},{y}): image from: ({image_start_x},{image_start_y}) to ({image_start_x+kernel_height},{image_start_y+kernel_width})")
            
#             sum = 0
#             NX = kernel_height // 2
#             NY = kernel_width // 2
#             for kx in range( -NX, NX+1):
#                 for ky in range( -NY, NY+1 ):
#                     rel_pos_in_kernel_x = kx + NX # x-i
#                     rel_pos_in_kernel_y = ky + NY # y-j
                    
#                     rel_pos_in_image_x = NX - kx # 2
#                     rel_pos_in_image_y = NY - ky # 2
                    
#                     act_pos_in_image_x = rel_pos_in_image_x + image_start_x # 2 + 2 = 4
#                     act_pos_in_image_y = rel_pos_in_image_y + image_start_y # 3 + 2 = 5
                    
#                     # if( rel_pos_in_kernel_x >= kernel_height or rel_pos_in_kernel_y >= kernel_width):
#                     #     print("Outside")
#                     #     print(rel_pos_in_kernel_x, rel_pos_in_kernel_y)
#                     #     print(kernel)
                    
#                     k_val = kernel[ rel_pos_in_kernel_x ][ rel_pos_in_kernel_y ]
#                     i_val = padded_image[ act_pos_in_image_x ][ act_pos_in_image_y ]
                    
#                     # if x == 1 and y == 2:
#                     #     #print(k_val, "*", i_val)
#                     #     print(f"({rel_pos_in_image_x}, {rel_pos_in_image_y}) * ({rel_pos_in_kernel_x}, {rel_pos_in_kernel_y}): {k_val} * {i_val} Actual pos in image: ({act_pos_in_image_x}, {act_pos_in_image_y})")
                    
#                     sum +=  k_val * i_val
#             output[x,y] = sum

#     # print("Output before cropping")
#     # print(output)
#     # Crop the output to the original image size
#     out = output[kernel_center[0]:-kernel_height + kernel_center[0] + 1, kernel_center[1]:-kernel_width + kernel_center[1] + 1]
    
#     return out

import numpy as np

def convolution(image,kernel,centerx=None,centery=None):

    kernel_height = kernel.shape[0]
    kernel_width = kernel.shape[1]

    newcenterx = centerx
    newcentery = centery

    
    centerx = kernel_height // 2
    centery = kernel_width // 2

    print(centerx,centery,newcenterx,newcentery)

    image = cv2.copyMakeBorder(src=image, top=newcenterx, bottom=kernel_height - newcenterx - 1, left=newcentery, right=kernel_width - newcentery -1,
                                       borderType=cv2.BORDER_CONSTANT)


    output_image = np.zeros((image.shape[0], image.shape[1]))

    for i in range(centerx,image.shape[0]-centerx):
        for j in range(centery,image.shape[1]-centery):
            sum = 0
            for k in range(kernel_height):
                for l in range(kernel_width):
                    sum += kernel[kernel_height-k-1][kernel_width-l-1] * image[i-centerx+k][j-centery+l] 
            # output_image[i][j] = sum
            a = i+newcenterx-centerx
            b = j+newcentery-centery
            output_image[a][b] = sum
            

    return(output_image)