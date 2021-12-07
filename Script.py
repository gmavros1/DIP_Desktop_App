import numpy as np
import matplotlib.pyplot as plt
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
    paddingHeight = (kernelHeight // 2)
    paddingWidth = (kernelWidth // 2)

    # padding for convolution
    paddedImage = np.zeros((imageHeight + 2 * paddingHeight, imageWidth + 2 * paddingWidth), dtype=np.int8)
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
    for i in range(paddingHeight):
        for j in range(paddingWidth + 1):
            B[i][j], B[paddingHeight - i + 1][paddingWidth - j + 1] = B[paddingHeight - i + 1][paddingWidth - j + 1], B[i][j]

    out = np.zeros((outputHeight, outputWidth), dtype=np.int8)

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


matrix = [[3, 23, 255, 6, 23, 87, 33, 54, 1, 8],
          [32, 67, 255, 65, 7, 81, 87, 52, 79, 23],
          [1, 23, 255, 6, 22, 87, 200, 54, 2, 8],
          [31, 23, 255, 255, 23, 87, 33, 54, 1, 8],
          [100, 23, 255, 6, 23, 87, 33, 54, 1, 8],
          [150, 12, 57, 112, 34, 25, 56, 12, 23, 56],
          [5, 4, 43, 255, 255, 255, 255, 255, 6, 2],
          [3, 23, 6, 6, 23, 87, 33, 54, 1, 8]]

kernel = [[-1, 0, 1],
          [-1, 0, 1],
          [-1, 0, 1]]

out = myConv2(matrix, kernel, 'pad')

plt.subplot(3, 1, 1)
plt.imshow(matrix, cmap='gray')
plt.subplot(3, 1, 2)
plt.imshow(kernel, cmap='gray')
plt.subplot(3, 1, 3)
plt.imshow(out, cmap='gray')
plt.show()

