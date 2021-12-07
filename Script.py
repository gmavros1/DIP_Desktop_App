import numpy
import numpy as np
import cv2


def sizeAfterConv(height_width, kernelHeight_Width, padTop, padBot, stride):
    return ((height_width + padTop + padBot - kernelHeight_Width) // stride) + 1


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
    kernelWidth = len(B[0])
    paddingHeight = (kernelHeight // 2) + 1
    paddingWidth = (kernelWidth // 2) + 1

    # padding for convolution
    paddedImage = np.zeros((imageHeight + 2 * paddingHeight, imageWidth + 2 * paddingWidth), dtype=numpy.int8)
    im = 0
    for i in range(paddingHeight, imageHeight):
        jm = 0
        for j in range(paddingWidth, imageWidth):
            paddedImage[i][j] = A[im][jm]
            jm += 1
        im += 1

    # define output size
    outputHeight = sizeAfterConv(imageHeight, kernelHeight, paddingHeight, paddingWidth, 1)
    outputWidth = sizeAfterConv(imageWidth, kernelWidth, paddingWidth, paddingWidth, 1)

    # flip kernel
    for i in range(paddingHeight):
        for j in range(paddingWidth):
            B[i][j], B[paddingHeight - i - 1][paddingWidth - j - 1] = B[paddingHeight - i - 1][paddingWidth - j - 1], \
                                                                      B[i][j]

    out = np.zeros((outputHeight, outputWidth), dtype=numpy.int8)

    for k in range(outputHeight):
        for l in range(outputWidth):
            for i in range(len(paddedImage) - kernelHeight):
                for j in range(len(paddedImage[0]) - kernelWidth):
                    convSum = 0
                    for m in range(kernelHeight):
                        for n in range(kernelWidth):
                            convSum += paddedImage[n + i][n + j] * B[m][n]
                    out[k][l] = convSum

    if param == 'same':
        for i in range(imageHeight):
            for j in range(imageWidth):
                A[i][j] = out[i + (outputHeight - imageHeight)][j + outputWidth - imageWidth]
        return np.array(A)
    else:
        return out


matrix = [[3, 23, 6, 6, 23, 87, 33, 54, 1, 8],
          [32, 67, 3, 65, 7, 81, 87, 52, 79, 23],
          [1, 23, 6, 6, 22, 87, 365, 54, 2, 8],
          [3, 23, 6, 6, 23, 87, 33, 54, 1, 8],
          [3, 23, 6, 6, 23, 87, 33, 54, 1, 8],
          [86, 12, 57, 112, 34, 25, 56, 12, 23, 56],
          [5, 4, 43, 798, 2, 8, 2, 9, 6, 2],
          [3, 23, 6, 6, 23, 87, 33, 54, 1, 8]]

kernel = [[-1, 0, 1],
          [-1, 0, 1],
          [-1, 0, 1]]

print(myConv2(matrix, kernel, 'pad'))
