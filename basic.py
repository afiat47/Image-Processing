import gausian as g
import mean as m
import conv as co
import laplacian as lp
import sobel as sb
import cv2
import numpy as np
import LOG


LinaImage = "D:\\project\\pythonProject\\Lena.jpg"

def normalize(image):
    cv2.normalize(image,image, 0, 255, cv2.NORM_MINMAX)
    image = np.round(image).astype(np.uint8)
    return image

def get_Sigma():
    print("Enter the sigma x and sigma y value respectively")
    sigmaX = float(input())
    sigmaY = float(input())
    return sigmaX, sigmaY

def get_dimention():
    print("Enter the Height and Width value respectively")
    h = int(input())
    w = int(input())
    return h, w

def get_laplacian_dimention():
    print("Enter ODD Height and ODD Width value respectively")
    h = int(input())
    w = int(input())
    return h, w


def get_Center():
    print('Enter center index for the kernel ')
    p = int(input())
    q = int(input())
    return p, q


def get_one_dimensional_gausian(sizex=None,sizey=None):
    if sizex is None:
        sizex=1
    if sizey is None:
        sizey=1
    sigmaX, sigmaY = get_Sigma()
    kernel = g.gaussian_kernel(sigmaX=sigmaX, sigmaY=sigmaY, sizeX=sizex ,sizeY = sizey)
    return kernel




def first_derivative_kernel_X(size):
    kernel = []
    n = size // 2
    for i in range(-n,n+1):
        kernel.append(i)
    
    return kernel


def first_derivative_kernel_Y(size):
    kernel = []
    n = size // 2
    for i in range(-n, n + 1):
        kernel.append([i])
    return kernel


def get_parameters_of_sobelX():
    print('Enter size of the first derivative kernel')
    size = int(input())
    kernel1 = first_derivative_kernel_X(size)
    print(kernel1)

    print('Enter height of the gaussian kernel')
    sizex = int(input())
    kernel2 = get_one_dimensional_gausian(sizex=sizex)

    return kernel1, kernel2


def get_parameters_of_sobelY():
    print('Enter size of the first derivative kernel')
    size = int(input())
    kernel1 = first_derivative_kernel_Y(size)
    print(kernel1)

    print('Enter width of the gaussian kernel')
    sizey = int(input())
    kernel2 = get_one_dimensional_gausian(sizey=sizey)

    return kernel1, kernel2



def get_LOG_Gausian():
    sigmaX, sigmaY = get_Sigma()
    kernel = g.gaussian_kernel(sigmaX=sigmaX, sigmaY=sigmaY)
    
    return kernel

def get_LOG_laplacian():
    height, width = get_laplacian_dimention()
    kernel = lp.laplacian_mask(height, width)
    return kernel

def gaussianFilter():
    print("Select \n 1 for grayscale image \n 2 for color \n 3 for hsv:")
    select = int(input())
    if select == 1:
        img = cv2.imread(LinaImage, cv2.IMREAD_GRAYSCALE)
        cv2.imshow("input", img)
        cv2.waitKey(0)        
        sigmaX, sigmaY = get_Sigma()
        kernel = g.gaussian_kernel(sigmaX=sigmaX, sigmaY=sigmaY)
        p , q = get_Center()
        out = co.convolution(img,kernel, p, q)
        out = normalize(out)
        print (out)
        cv2.imshow("output", out)

    elif select == 2:
        img = cv2.imread(LinaImage)
        cv2.imshow("input", img)  
        cv2.waitKey(0)           
        b1, g1, r1 = cv2.split(img)
        sigmaX, sigmaY = get_Sigma()
        kernel = g.gaussian_kernel(sigmaX=sigmaX, sigmaY=sigmaY)
        p , q = get_Center()

        cv2.imshow("blue",b1)
        b1 = co.convolution(image=b1,kernel=kernel,centerx= p,centery= q)
        newb1 = b1.copy()
        newb1 = normalize(newb1)
        cv2.imshow("convolution blue",newb1)
        cv2.waitKey(0)

        cv2.imshow("red",r1)
        r1 = co.convolution(image=r1,kernel=kernel,centerx= p,centery= q)
        newr1 = r1.copy()
        newr1 = normalize(newr1)
        cv2.imshow("convolution red",newr1)
        cv2.waitKey(0) 

        cv2.imshow("green",g1)
        g1 = co.convolution(image=g1,kernel=kernel,centerx= p,centery= q)
        newg1 = g1.copy()
        newg1 = normalize(newg1)
        cv2.imshow("convolution green",newg1)
        cv2.waitKey(0)
        merged = cv2.merge((b1, g1, r1))

        out = normalize(merged)
        print (out)
        cv2.imshow("output", out)

    else:
        img = cv2.imread(LinaImage)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        cv2.imshow("input in HSV formet", img)  
        cv2.waitKey(0)  
        b1, g1, r1 = cv2.split(img)
        sigmaX, sigmaY = get_Sigma()
        kernel = g.gaussian_kernel(sigmaX=sigmaX, sigmaY=sigmaY)
        p , q = get_Center()


        cv2.imshow("HUE",b1)
        b1 = co.convolution(image=b1,kernel=kernel,centerx= p,centery= q)
        newb1 = b1.copy()
        newb1 = normalize(newb1)
        cv2.imshow("convolution HUE",newb1)
        cv2.waitKey(0)

        cv2.imshow("Saturation",g1)
        g1 = co.convolution(image=g1,kernel=kernel,centerx= p,centery= q)
        newg1 = g1.copy()
        newg1 = normalize(newg1)
        cv2.imshow("convolution Saturation",newg1)
        cv2.waitKey(0)

        cv2.imshow("Value",r1)
        r1 = co.convolution(image=r1,kernel=kernel,centerx= p,centery= q)
        newr1 = r1.copy()
        newr1 = normalize(newr1)
        cv2.imshow("convolution Value",newr1)
        cv2.waitKey(0)

        merged = cv2.merge((b1, g1, r1))
        out = normalize(merged)
        print (out)
        cv2.imshow("Merged output HSV picture", out)
        cv2.waitKey(0)

        BacktoRGB = cv2.cvtColor(out, cv2.COLOR_HSV2RGB)
        out = normalize(BacktoRGB)
        cv2.imshow("HSV brought back to RGB,(Convolution in HSV space)", out)
        cv2.waitKey(0)

        # now conv in RGB space for this image.     
        img2 = cv2.imread(LinaImage) 
        tb1, tg1, tr1 = cv2.split(img2)
        tb1 = co.convolution(image=tb1,kernel=kernel,centerx= p,centery= q)
        tg1 = co.convolution(image=tg1,kernel=kernel,centerx= p,centery= q)
        tr1 = co.convolution(image=tr1,kernel=kernel,centerx= p,centery= q)
        newly = cv2.merge((tb1, tg1, tr1))
        out2 = normalize(newly)
        print (out2)
        cv2.imshow("Convolution in RGB space", out2)


        # nedd to defferenciate every seperate channel then merge
        diff = out2 - out
        out = normalize(diff)
        cv2.imshow("Difference between New_RGB and Merged_HSV", out)
        cv2.waitKey(0)

    return 0




def meanFilter():
    print("Select \n 1 for grayscale image \n 2 for color \n 3 for hsv:")
    select = int(input())
    if select == 1:
        img = cv2.imread(LinaImage, cv2.IMREAD_GRAYSCALE)
        cv2.imshow("input", img)
        cv2.waitKey(0)        
        height, width = get_dimention()
        kernel = m.mean(height, width)
        p , q = get_Center()
        out = co.convolution(img,kernel, p, q)
        out = normalize(out)
        print (out)
        cv2.imshow("output", out)

    elif select == 2:
        img = cv2.imread(LinaImage)
        cv2.imshow("input", img)  
        cv2.waitKey(0)           
        b1, g1, r1 = cv2.split(img)
        height, width = get_dimention()
        kernel = m.mean(height, width)
        p , q = get_Center()

        cv2.imshow("blue",b1)
        b1 = co.convolution(image=b1,kernel=kernel,centerx= p,centery= q)
        newb1 = b1.copy()
        newb1 = normalize(newb1)
        cv2.imshow("convolution blue",newb1)
        cv2.waitKey(0)

        cv2.imshow("red",r1)
        r1 = co.convolution(image=r1,kernel=kernel,centerx= p,centery= q)
        newr1 = r1.copy()
        newr1 = normalize(newr1)
        cv2.imshow("convolution red",newr1)
        cv2.waitKey(0) 

        cv2.imshow("green",g1)
        g1 = co.convolution(image=g1,kernel=kernel,centerx= p,centery= q)
        newg1 = g1.copy()
        newg1 = normalize(newg1)
        cv2.imshow("convolution green",newg1)
        cv2.waitKey(0)
        merged = cv2.merge((b1, g1, r1))

        out = normalize(merged)
        print (out)
        cv2.imshow("output", out)

    else:
        img = cv2.imread(LinaImage)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        cv2.imshow("input in HSV formet", img)  
        cv2.waitKey(0)  
        b1, g1, r1 = cv2.split(img)
        height, width = get_dimention()
        kernel = m.mean(height, width)
        p , q = get_Center()


        cv2.imshow("HUE",b1)
        b1 = co.convolution(image=b1,kernel=kernel,centerx= p,centery= q)
        newb1 = b1.copy()
        newb1 = normalize(newb1)
        cv2.imshow("convolution HUE",newb1)
        cv2.waitKey(0)

        cv2.imshow("Saturation",g1)
        g1 = co.convolution(image=g1,kernel=kernel,centerx= p,centery= q)
        newg1 = g1.copy()
        newg1 = normalize(newg1)
        cv2.imshow("convolution Saturation",newg1)
        cv2.waitKey(0)

        cv2.imshow("Value",r1)
        r1 = co.convolution(image=r1,kernel=kernel,centerx= p,centery= q)
        newr1 = r1.copy()
        newr1 = normalize(newr1)
        cv2.imshow("convolution Value",newr1)
        cv2.waitKey(0)

        merged = cv2.merge((b1, g1, r1))
        out = normalize(merged)
        print (out)
        cv2.imshow("Merged output HSV picture", out)
        cv2.waitKey(0)

        BacktoRGB = cv2.cvtColor(out, cv2.COLOR_HSV2RGB)
        out = normalize(BacktoRGB)
        cv2.imshow("HSV brought back to RGB,(Convolution in HSV space)", out)
        cv2.waitKey(0)

        # now conv in RGB space for this image.     
        img2 = cv2.imread(LinaImage) 
        tb1, tg1, tr1 = cv2.split(img2)
        tb1 = co.convolution(image=tb1,kernel=kernel,centerx= p,centery= q)
        tg1 = co.convolution(image=tg1,kernel=kernel,centerx= p,centery= q)
        tr1 = co.convolution(image=tr1,kernel=kernel,centerx= p,centery= q)
        newly = cv2.merge((tb1, tg1, tr1))
        out2 = normalize(newly)
        print (out2)
        cv2.imshow("Convolution in RGB space", out2)


        # nedd to defferenciate every seperate channel then merge
        diff = out2 - out
        out = normalize(diff)
        cv2.imshow("Difference between New_RGB and Merged_HSV", out)
        cv2.waitKey(0)

    return 0





def laplacianFilter():
    print("Select \n 1 for grayscale image \n 2 for color \n 3 for hsv:")
    select = int(input())
    if select == 1:
        img = cv2.imread(LinaImage, cv2.IMREAD_GRAYSCALE)
        cv2.imshow("input", img)
        cv2.waitKey(0)        
        height, width = get_laplacian_dimention()
        kernel = lp.laplacian_mask(height, width)
        p , q = get_Center()
        out = co.convolution(img,kernel, p, q)
        out = normalize(out)
        print (out)
        cv2.imshow("output", out)

    elif select == 2:
        img = cv2.imread(LinaImage)
        cv2.imshow("input", img)  
        cv2.waitKey(0)           
        b1, g1, r1 = cv2.split(img)
        height, width = get_laplacian_dimention()
        kernel = lp.laplacian_mask(height, width)
        p , q = get_Center()

        cv2.imshow("blue",b1)
        b1 = co.convolution(image=b1,kernel=kernel,centerx= p,centery= q)
        newb1 = b1.copy()
        newb1 = normalize(newb1)
        cv2.imshow("convolution blue",newb1)
        cv2.waitKey(0)

        cv2.imshow("red",r1)
        r1 = co.convolution(image=r1,kernel=kernel,centerx= p,centery= q)
        newr1 = r1.copy()
        newr1 = normalize(newr1)
        cv2.imshow("convolution red",newr1)
        cv2.waitKey(0) 

        cv2.imshow("green",g1)
        g1 = co.convolution(image=g1,kernel=kernel,centerx= p,centery= q)
        newg1 = g1.copy()
        newg1 = normalize(newg1)
        cv2.imshow("convolution green",newg1)
        cv2.waitKey(0)
        merged = cv2.merge((b1, g1, r1))

        out = normalize(merged)
        print (out)
        cv2.imshow("output", out)

    else:
        img = cv2.imread(LinaImage)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        cv2.imshow("input in HSV formet", img)  
        cv2.waitKey(0)  
        b1, g1, r1 = cv2.split(img)
        height, width = get_laplacian_dimention()
        kernel = lp.laplacian_mask(height, width)
        p , q = get_Center()


        cv2.imshow("HUE",b1)
        b1 = co.convolution(image=b1,kernel=kernel,centerx= p,centery= q)
        newb1 = b1.copy()
        newb1 = normalize(newb1)
        cv2.imshow("convolution HUE",newb1)
        cv2.waitKey(0)

        cv2.imshow("Saturation",g1)
        g1 = co.convolution(image=g1,kernel=kernel,centerx= p,centery= q)
        newg1 = g1.copy()
        newg1 = normalize(newg1)
        cv2.imshow("convolution Saturation",newg1)
        cv2.waitKey(0)

        cv2.imshow("Value",r1)
        r1 = co.convolution(image=r1,kernel=kernel,centerx= p,centery= q)
        newr1 = r1.copy()
        newr1 = normalize(newr1)
        cv2.imshow("convolution Value",newr1)
        cv2.waitKey(0)

        merged = cv2.merge((b1, g1, r1))
        out = normalize(merged)
        print (out)
        cv2.imshow("Merged output HSV picture", out)
        cv2.waitKey(0)

        BacktoRGB = cv2.cvtColor(out, cv2.COLOR_HSV2RGB)
        out = normalize(BacktoRGB)
        cv2.imshow("HSV brought back to RGB,(Convolution in HSV space)", out)
        cv2.waitKey(0)

        # now conv in RGB space for this image.     
        img2 = cv2.imread(LinaImage) 
        tb1, tg1, tr1 = cv2.split(img2)
        tb1 = co.convolution(image=tb1,kernel=kernel,centerx= p,centery= q)
        tg1 = co.convolution(image=tg1,kernel=kernel,centerx= p,centery= q)
        tr1 = co.convolution(image=tr1,kernel=kernel,centerx= p,centery= q)
        newly = cv2.merge((tb1, tg1, tr1))
        out2 = normalize(newly)
        print (out2)
        cv2.imshow("Convolution in RGB space", out2)


        # nedd to defferenciate every seperate channel then merge
        diff = out2 - out
        out = normalize(diff)
        cv2.imshow("Difference between New_RGB and Merged_HSV", out)
        cv2.waitKey(0)

    return 0




def laplacianOfGaussianFilter():
    print("Select \n 1 for grayscale image \n 2 for color \n 3 for hsv:")
    select = int(input())
    if select == 1:
        img = cv2.imread(LinaImage, cv2.IMREAD_GRAYSCALE)
        cv2.imshow("input", img)
        cv2.waitKey(0)        
        gausian = get_LOG_Gausian()
        laplacian = get_LOG_laplacian()
        kernel = LOG.LOG(gausian, laplacian)
        p , q = get_Center()
        out = co.convolution(img,kernel, p, q)
        out = normalize(out)
        print (out)
        cv2.imshow("output", out)

    elif select == 2:
        img = cv2.imread(LinaImage)
        cv2.imshow("input", img)  
        cv2.waitKey(0)           
        b1, g1, r1 = cv2.split(img)
        gausian = get_LOG_Gausian()
        laplacian = get_LOG_laplacian()
        kernel = LOG.LOG(gausian, laplacian)
        p , q = get_Center()

        cv2.imshow("blue",b1)
        b1 = co.convolution(image=b1,kernel=kernel,centerx= p,centery= q)
        newb1 = b1.copy()
        newb1 = normalize(newb1)
        cv2.imshow("convolution blue",newb1)
        cv2.waitKey(0)

        cv2.imshow("red",r1)
        r1 = co.convolution(image=r1,kernel=kernel,centerx= p,centery= q)
        newr1 = r1.copy()
        newr1 = normalize(newr1)
        cv2.imshow("convolution red",newr1)
        cv2.waitKey(0) 

        cv2.imshow("green",g1)
        g1 = co.convolution(image=g1,kernel=kernel,centerx= p,centery= q)
        newg1 = g1.copy()
        newg1 = normalize(newg1)
        cv2.imshow("convolution green",newg1)
        cv2.waitKey(0)
        merged = cv2.merge((b1, g1, r1))

        out = normalize(merged)
        print (out)
        cv2.imshow("output", out)

    else:
        img = cv2.imread(LinaImage)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        cv2.imshow("input in HSV formet", img)  
        cv2.waitKey(0)  
        b1, g1, r1 = cv2.split(img)
        gausian = get_LOG_Gausian()
        laplacian = get_LOG_laplacian()
        kernel = LOG.LOG(gausian, laplacian)
        p , q = get_Center()


        cv2.imshow("HUE",b1)
        b1 = co.convolution(image=b1,kernel=kernel,centerx= p,centery= q)
        newb1 = b1.copy()
        newb1 = normalize(newb1)
        cv2.imshow("convolution HUE",newb1)
        cv2.waitKey(0)

        cv2.imshow("Saturation",g1)
        g1 = co.convolution(image=g1,kernel=kernel,centerx= p,centery= q)
        newg1 = g1.copy()
        newg1 = normalize(newg1)
        cv2.imshow("convolution Saturation",newg1)
        cv2.waitKey(0)

        cv2.imshow("Value",r1)
        r1 = co.convolution(image=r1,kernel=kernel,centerx= p,centery= q)
        newr1 = r1.copy()
        newr1 = normalize(newr1)
        cv2.imshow("convolution Value",newr1)
        cv2.waitKey(0)

        merged = cv2.merge((b1, g1, r1))
        out = normalize(merged)
        print (out)
        cv2.imshow("Merged output HSV picture", out)
        cv2.waitKey(0)

        BacktoRGB = cv2.cvtColor(out, cv2.COLOR_HSV2RGB)
        out = normalize(BacktoRGB)
        cv2.imshow("HSV brought back to RGB,(Convolution in HSV space)", out)
        cv2.waitKey(0)

        # now conv in RGB space for this image.     
        img2 = cv2.imread(LinaImage) 
        tb1, tg1, tr1 = cv2.split(img2)
        tb1 = co.convolution(image=tb1,kernel=kernel,centerx= p,centery= q)
        tg1 = co.convolution(image=tg1,kernel=kernel,centerx= p,centery= q)
        tr1 = co.convolution(image=tr1,kernel=kernel,centerx= p,centery= q)
        newly = cv2.merge((tb1, tg1, tr1))
        out2 = normalize(newly)
        print (out2)
        cv2.imshow("Convolution in RGB space", out2)


        # nedd to defferenciate every seperate channel then merge
        diff = out2 - out
        out = normalize(diff)
        cv2.imshow("Difference between New_RGB and Merged_HSV", out)
        cv2.waitKey(0)

    return 0





def sobelFilter():
    print("Select \n 1 for grayscale image \n 2 for color \n 3 for hsv:")
    select = int(input())
    if select == 1:
        img = cv2.imread(LinaImage, cv2.IMREAD_GRAYSCALE)
        cv2.imshow("input", img)
        cv2.waitKey(0)

        print('Select \n 1 Vertical sharpening \n 2 Horizontal Sharpening:')
        choice = int(input())
        if choice == 1:
            k1, k2 = get_parameters_of_sobelX()
            kernel = sb.sobel(k1, k2)
        else:
            k1, k2 = get_parameters_of_sobelY()
            kernel = sb.sobel2(k1, k2)
        p , q = get_Center()
        out = co.convolution(img,kernel, p, q)
        out = normalize(out)
        print (out)
        cv2.imshow("output", out)

    elif select == 2:
        img = cv2.imread(LinaImage)
        cv2.imshow("input", img)  
        cv2.waitKey(0)           
        b1, g1, r1 = cv2.split(img)
        print('Select \n 1 Vertical sharpening \n 2 Horizontal Sharpening:')
        choice = int(input())
        if choice == 1:
            k1, k2 = get_parameters_of_sobelX()
            kernel = sb.sobel(k1, k2)
        else:
            k1, k2 = get_parameters_of_sobelY()
            kernel = sb.sobel2(k1, k2)
        p , q = get_Center()

        cv2.imshow("blue",b1)
        b1 = co.convolution(image=b1,kernel=kernel,centerx= p,centery= q)
        newb1 = b1.copy()
        newb1 = normalize(newb1)
        cv2.imshow("convolution blue",newb1)
        cv2.waitKey(0)

        cv2.imshow("red",r1)
        r1 = co.convolution(image=r1,kernel=kernel,centerx= p,centery= q)
        newr1 = r1.copy()
        newr1 = normalize(newr1)
        cv2.imshow("convolution red",newr1)
        cv2.waitKey(0) 

        cv2.imshow("green",g1)
        g1 = co.convolution(image=g1,kernel=kernel,centerx= p,centery= q)
        newg1 = g1.copy()
        newg1 = normalize(newg1)
        cv2.imshow("convolution green",newg1)
        cv2.waitKey(0)
        merged = cv2.merge((b1, g1, r1))

        out = normalize(merged)
        print (out)
        cv2.imshow("output", out)

    else:
        img = cv2.imread(LinaImage)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        cv2.imshow("input in HSV formet", img)  
        cv2.waitKey(0)  
        b1, g1, r1 = cv2.split(img)
        print('Select \n 1 Vertical sharpening \n 2 Horizontal Sharpening:')
        choice = int(input())
        if choice == 1:
            k1, k2 = get_parameters_of_sobelX()
            kernel = sb.sobel(k1, k2)
        else:
            k1, k2 = get_parameters_of_sobelY()
            kernel = sb.sobel2(k1, k2)
        p , q = get_Center()


        cv2.imshow("HUE",b1)
        b1 = co.convolution(image=b1,kernel=kernel,centerx= p,centery= q)
        newb1 = b1.copy()
        newb1 = normalize(newb1)
        cv2.imshow("convolution HUE",newb1)
        cv2.waitKey(0)

        cv2.imshow("Saturation",g1)
        g1 = co.convolution(image=g1,kernel=kernel,centerx= p,centery= q)
        newg1 = g1.copy()
        newg1 = normalize(newg1)
        cv2.imshow("convolution Saturation",newg1)
        cv2.waitKey(0)

        cv2.imshow("Value",r1)
        r1 = co.convolution(image=r1,kernel=kernel,centerx= p,centery= q)
        newr1 = r1.copy()
        newr1 = normalize(newr1)
        cv2.imshow("convolution Value",newr1)
        cv2.waitKey(0)

        merged = cv2.merge((b1, g1, r1))
        out = normalize(merged)
        print (out)
        cv2.imshow("Merged output HSV picture", out)
        cv2.waitKey(0)

        BacktoRGB = cv2.cvtColor(out, cv2.COLOR_HSV2RGB)
        out = normalize(BacktoRGB)
        cv2.imshow("HSV brought back to RGB,(Convolution in HSV space)", out)
        cv2.waitKey(0)

        # now conv in RGB space for this image.     
        img2 = cv2.imread(LinaImage) 
        tb1, tg1, tr1 = cv2.split(img2)
        tb1 = co.convolution(image=tb1,kernel=kernel,centerx= p,centery= q)
        tg1 = co.convolution(image=tg1,kernel=kernel,centerx= p,centery= q)
        tr1 = co.convolution(image=tr1,kernel=kernel,centerx= p,centery= q)
        newly = cv2.merge((tb1, tg1, tr1))
        out2 = normalize(newly)
        print (out2)
        cv2.imshow("Convolution in RGB space", out2)


        # nedd to defferenciate every seperate channel then merge
        diff = out2 - out
        out = normalize(diff)
        cv2.imshow("Difference between New_RGB and Merged_HSV", out)
        cv2.waitKey(0)

    return 0





def Main():
    while True:
        print("Select the type of filter you want: ")
        print("1  Gaussian filter")
        print("2  Mean Filter")
        print("3  Laplacian Filter")
        print("4  LoG Filter")
        print("5  Sobel Filter")

        choise = int(input())
        if choise == 1:
            gaussianFilter()
        elif choise == 2:
            meanFilter()
        elif choise == 3:
            laplacianFilter()
        elif choise == 4:
            laplacianOfGaussianFilter()
        else:
            sobelFilter()
        cv2.waitKey(0)
        cv2.destroyAllWindows()

Main()