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
    paddingHeight = WindowHeight - 1
    paddingWidth = WindowWidth - 1

    # padding for convolution
    paddedImage = np.zeros((imageHeight + 2 * paddingHeight, imageWidth + 2 * paddingWidth), dtype=int)
    im = 0
    for i in range(paddingHeight, imageHeight + paddingHeight):
        jm = 0
        for j in range(paddingWidth, imageWidth + paddingWidth):
            paddedImage[i][j] = A[im][jm]
            jm += 1
        im += 1

    return paddedImage


def myConv2(A, B, param):
    imageHeight = len(A)
    imageWidth = len(A[0])
    kernelHeight = len(B)
    kernelWidth = len(B[0])
    paddingHeight = kernelHeight - 1
    paddingWidth = kernelWidth - 1

    # padding for convolution
    paddedImage = paddImage(A, kernelHeight, kernelWidth)

    # define output size
    outputHeight = len(paddedImage) - kernelHeight
    outputWidth = len(paddedImage[0]) - kernelWidth

    # flip kernel
    B = np.fliplr(B)
    B = np.flipud(B)

    #out = np.zeros(outputHeight * outputWidth, dtype=int)
    out = []

    convSum = 0
    for i in range(len(paddedImage) - kernelHeight):
        for j in range(len(paddedImage[0]) - kernelWidth):
            for m in range(kernelHeight):
                for n in range(kernelWidth):
                    convSum += float(paddedImage[m + i][n + j]) * B[m][n]
            out.append(int(convSum))
            convSum = 0

    out = np.array(out)
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
    windowHeight = 5
    windowWidth = 5
    paddedIm = paddImage(A, windowHeight, windowWidth)
    outputHeight = len(paddedIm) - windowHeight
    outputWidth = len(paddedIm[0]) - windowWidth
    # medianImage = np.zeros(outputHeight * outputWidth, dtype=int)  # 1D **** THE OUTPUT
    medianImage = []
    for i in range(len(paddedIm) - windowHeight):
        for j in range(len(paddedIm[0]) - windowWidth):
            temp = []
            for m in range(windowHeight):
                for n in range(windowWidth):
                    temp.append(paddedIm[m + i][n + j])
            temp = np.array(temp)
            medianImage.append(findMedian(temp))

    medianImage = np.array(medianImage)
    out2D = np.reshape(medianImage, (outputHeight, -1))  # maybe output height second parameter
    #return out2D
    return cropOutput(out2D, len(A), len(A[0]), outputHeight, outputWidth)

def medianTOconv(A, k):
    windowHeight = len(k)
    windowWidth = len(k[0])
    paddedIm = paddImage(A, windowHeight, windowWidth)
    outputHeight = len(paddedIm) - windowHeight
    outputWidth = len(paddedIm[0]) - windowWidth
    # medianImage = np.zeros(outputHeight * outputWidth, dtype=int)  # 1D **** THE OUTPUT
    medianImage = []
    k1D = np.reshape(k, (1, -1))
    for i in range(len(paddedIm) - windowHeight):
        for j in range(len(paddedIm[0]) - windowWidth):
            temp = []
            for m in range(windowHeight):
                for n in range(windowWidth):
                    temp.append(paddedIm[m + i][n + j])
            temp = np.array(temp)
            mul = temp * k1D
            medianImage.append(mul.sum())

    medianImage = np.array(medianImage)
    out2D = np.reshape(medianImage, (outputHeight, -1))  # maybe output height second parameter
    #return out2D
    return cropOutput(out2D, len(A), len(A[0]), outputHeight, outputWidth)

def myImFilter(A, param):
    if param == "mean":
        height = 8
        width = 8
        kernelF = np.full((height, width), (1/(height*width)))
        kernelF = np.reshape(kernelF, (height, -1))
        #kernelF = np.array([[1.0 / 9, 1.0 / 9, 1.0 / 9],
        #                    [1.0 / 9, 1.0 / 9, 1.0 / 9],
        #                    [1.0 / 9, 1.0 / 9, 1.0 / 9]])
        outMean = medianTOconv(A, kernelF)
        return outMean
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

#out = myConv2(matrix, kernel, 'same')
#plt.subplot(2, 3, 5)
#plt.imshow(out, cmap='gray')

#out = medianTOconv(matrix, kernel)
#plt.subplot(2, 3, 5)
#plt.imshow(out, cmap='gray')


# out1 = signal.convolve2d(matrix, kernel, boundary='symm', mode='same')
# plt.subplot(2, 3, 4)
# plt.imshow(out1, cmap='gray')


matrix = myImNoise(matrix, "salt")
plt.subplot(2, 3, 2)
plt.imshow(matrix, cmap='gray')

plt.subplot(2, 3, 6)
plt.imshow(myImFilter(matrix, "mean"), cmap='gray')

plt.show()
