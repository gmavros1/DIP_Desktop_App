import random
import numpy as np
import matplotlib.pyplot as plt
import cv2


def cropOutput(out2D, imageHeight, imageWidth, outputHeight, outputWidth):
    A = np.zeros((imageHeight, imageWidth), dtype=int)
    for i in range(imageHeight):
        for j in range(imageWidth):
            A[i][j] = out2D[i + (outputHeight - imageHeight)][j + outputWidth - imageWidth]

    return A


def sizeAfterConv(height_width, kernelHeight_Width, padTop, padBot, stride):
    return ((height_width + padTop + padBot - kernelHeight_Width) // stride) + 1


def paddImage(A, WindowHeight, WindowWidth):
    imageHeight = len(A)
    imageWidth = len(A[0])
    paddingHeight = (WindowHeight // 2)
    paddingWidth = (WindowWidth // 2)

    # padding for convolution
    paddedImage = np.zeros((imageHeight + 2 * paddingHeight, imageWidth + 2 * paddingWidth), dtype=int)
    im = 0
    for i in range(paddingHeight, imageHeight + 1):
        jm = 0
        for j in range(paddingWidth, imageWidth + 1):
            paddedImage[i][j] = A[im][jm]
            jm += 1
        im += 1

    return paddedImage


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
    paddedImage = paddImage(A, kernelHeight, kernelWidth)

    # define output size
    outputHeight = sizeAfterConv(imageHeight, kernelHeight, paddingHeight, paddingHeight, 1)
    outputWidth = sizeAfterConv(imageWidth, kernelWidth, paddingWidth, paddingWidth, 1)

    # flip kernel
    B = np.fliplr(B)
    B = np.flipud(B)

    out = np.zeros(outputHeight * outputWidth, dtype=int)

    count = 0
    convSum = 0
    for i in range(len(paddedImage) - kernelHeight + 1):
        for j in range(len(paddedImage[0]) - kernelWidth + 1):
            for m in range(kernelHeight):
                for n in range(kernelWidth):
                    convSum += float(paddedImage[m + i][n + j]) * B[m][n]
            out[count] = int(convSum)
            convSum = 0
            count += 1

    out2D = np.reshape(out, (outputHeight, -1))

    if param == 'same':
        outCropped = cropOutput(out2D, imageHeight, imageWidth, outputHeight, outputWidth)
        return np.array(outCropped)
    else:
        return out2D


def gaussian(A):
    noise = []
    height = len(A)
    width = len(A[0])
    for i in range(height):
        noise.append([])
        for j in range(width):
            noise[i].append(random.randint(-150, 150))

    noise = np.array(noise)
    A = A + noise
    return A


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
    if param == "gaussian":
        return gaussian(A)
    else:
        return saltAndPaper(A)


def findMedian(W):
    # w1D = np.reshape(W, (height * width, -1))
    w1Dsorted = np.sort(W)
    w1Dunique = np.unique(w1Dsorted)
    # print(w1Dunique)
    median = w1Dunique[len(w1Dunique) // 2]

    return median


def median(A):
    windowHeight = 3
    windowWidth = 3
    paddingHeight = (windowHeight // 2)
    paddingWidth = (windowWidth // 2)
    outputHeight = sizeAfterConv(len(A), windowHeight, paddingHeight, paddingHeight, 1)
    outputWidth = sizeAfterConv(len(A[0]), windowWidth, paddingWidth, paddingWidth, 1)
    paddedIm = paddImage(A, windowHeight, windowWidth)
    medianImage = np.zeros(outputHeight * outputWidth, dtype=int)  # 1D **** THE OUTPUT
    count = 0
    for i in range(len(paddedIm) - windowHeight + 1):
        for j in range(len(paddedIm[0]) - windowWidth + 1):
            temp = []
            for m in range(windowHeight):
                for n in range(windowWidth):
                    temp.append(paddedIm[m + i][n + j])
            temp = np.array(temp)
            medianImage[count] = findMedian(temp)
            count += 1
    out2D = np.reshape(medianImage, (outputHeight, -1))
    return cropOutput(out2D, len(A), len(A[0]), outputHeight, outputWidth)


def myImFilter(A, param):
    if param == "mean":
        kernelF = np.array([[1.0 / 9, 1.0 / 9, 1.0 / 9],
                           [1.0 / 9, 1.0 / 9, 1.0 / 9],
                           [1.0 / 9, 1.0 / 9, 1.0 / 9]])
        return myConv2(A, kernelF, "pad")
    else:
        return median(A)


matrix = cv2.imread('test.jpg', 0)  # read image - black and white
plt.subplot(2, 3, 1)
plt.imshow(matrix, cmap='gray')

kernel = np.array([[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]])

plt.subplot(2, 3, 3)
plt.imshow(kernel, cmap='gray')

# out = myConv2(matrix, kernel, 'same')
# plt.subplot(2, 3, 5)
# plt.imshow(out, cmap='gray')

# out1 = signal.convolve2d(matrix, kernel, boundary='symm', mode='same')
# plt.subplot(2, 3, 4)
# plt.imshow(out1, cmap='gray')


matrix = myImNoise(matrix, "salt")
plt.subplot(2, 3, 2)
plt.imshow(matrix, cmap='gray')

plt.subplot(2, 3, 6)
plt.imshow(myImFilter(matrix, "median"), cmap='gray')

plt.show()
