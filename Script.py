import numpy as np
import cv2


def sizeAfterConv(height_width, kernelHeight_Width, padTop, padBot, stride):
    return ((height_width + padTop + padBot - kernelHeight_Width) / stride) + 1


'''
A for image
B for kernel
param --> same : output same size as input
param --> pad : output size is padded
'''
def myConv2(A, B, param):
    imageHeight = len(A)
    imageWidth = len(A[0])
    kernelHeight = len(B)
    kernelWidth = len(A[B])
    paddingHeight = kernelHeight//2
    paddingWidth = kernelWidth//2

    outputHeight = sizeAfterConv(imageHeight, kernelHeight, paddingHeight, paddingWidth, 1)
    outputWidth = sizeAfterConv(imageWidth, kernelWidth, paddingWidth, paddingWidth, 1)

    np.zeros((outputHeight, outputWidth), dtype=float)



    print('Hello')


def readImage(imageName):
    A = cv2.imread(imageName)
    cv2.imshow('image', A)  # show image
    cv2.waitKey(0)  # wait for key press
    cv2.destroyAllWindows()  # close image window
