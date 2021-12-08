import random

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import cv2


def sizeAfterConv(height_width, kernelHeight_Width, padTop, padBot, stride):
    return ((height_width + padTop + padBot - kernelHeight_Width) // stride) + 1


def myConv2(A, B, param):
    """
    A for image
    B for kernel
    param --> same : output same size as input
    param --> pad : output size is padded
    """
    imageHeight = len(A)
    imageWidth = len(A[0])
    kernelHeight = len(B)
    kernelWidth = len(B[0])
    paddingHeight = (kernelHeight // 2)
    paddingWidth = (kernelWidth // 2)

    # padding for convolution
    paddedImage = np.zeros((imageHeight + 2 * paddingHeight, imageWidth + 2 * paddingWidth), dtype=int)
    im = 0
    for i in range(paddingHeight, imageHeight + 1):
        jm = 0
        for j in range(paddingWidth, imageWidth + 1):
            paddedImage[i][j] = A[im][jm]
            jm += 1
        im += 1

    # define output size
    outputHeight = sizeAfterConv(imageHeight, kernelHeight, paddingHeight, paddingWidth, 1)
    outputWidth = sizeAfterConv(imageWidth, kernelWidth, paddingWidth, paddingWidth, 1)

    # flip kernel
    B = np.fliplr(B)
    B = np.flipud(B)

    out = np.zeros(outputHeight * outputWidth, dtype=int)

    count = 0
    for i in range(len(paddedImage) - kernelHeight + 1):
        for j in range(len(paddedImage[0]) - kernelWidth + 1):
            convSum = 0
            for m in range(kernelHeight):
                for n in range(kernelWidth):
                    convSum += paddedImage[m + i][n + j] * B[m][n]
            out[count] = convSum
            count += 1

    out2D = np.reshape(out, (outputHeight, -1))

    if param == 'same':
        for i in range(imageHeight):
            for j in range(imageWidth):
                A[i][j] = out2D[i + (outputHeight - imageHeight)][j + outputWidth - imageWidth]
        return np.array(A)
    else:
        return out2D


def saltAndPaper(A):
    for i in range(len(A)):
        for j in range(len(A[0])):
            if random.uniform(0.0, 1.0) < 0.5:  # random probability
                if random.uniform(0.0, 1.0) < 0.5:
                    A[i][j] = 0
                else:
                    A[i][j] = 254

    return A


def myImNoise(A, param):

    return saltAndPaper(A)


'''height = 128
width = 160
matrix = []
for i in range(height):
    matrix.append([])
    for j in range(width):
        matrix[i].append(random.randint(0, 255))

matrix = np.array(matrix) '''

'''matrix = np.array([[3, 23, 255, 6, 23, 87, 33, 54, 1, 8],
                   [32, 67, 255, 65, 7, 81, 87, 52, 79, 23],
                   [1, 23, 255, 6, 22, 87, 200, 54, 2, 8],
                   [31, 23, 255, 255, 23, 87, 33, 54, 1, 8],
                   [100, 23, 255, 6, 23, 87, 33, 54, 1, 8],
                   [150, 12, 57, 112, 34, 25, 56, 12, 23, 56],
                   [5, 4, 43, 255, 255, 255, 255, 255, 6, 2],
                   [3, 23, 6, 6, 23, 87, 33, 54, 1, 8]])'''


matrix = cv2.imread('test.jpg', 0)  # read image - black and white

matrix = myImNoise(matrix, "salt")

plt.subplot(1, 1, 1)
plt.imshow(matrix, cmap='gray')

#kernel = np.array([[-1, 0, 1],
#                   [-1, 0, 1],
#                   [-1, 0, 1]])

#plt.subplot(2, 2, 2)
#plt.imshow(kernel, cmap='gray')

#out1 = signal.convolve2d(matrix, kernel, boundary='symm', mode='same')
#plt.subplot(2, 2, 4)
#plt.imshow(out1, cmap='gray')

#out = myConv2(matrix, kernel, 'same')
#plt.subplot(2, 2, 3)
#plt.imshow(out, cmap='gray')

plt.show()
