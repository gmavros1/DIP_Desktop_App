import random
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import cv2


class App:
    def __init__(self):
        self.initialImage = None
        self.processedImage = None
        self.imageHeight = None
        self.imageWidth = None

    def addImage(self, img):
        """array style"""
        self.initialImage = img
        self.processedImage = img
        self.imageHeight = len(self.initialImage)
        self.imageWidth = len(self.initialImage[0])


    @staticmethod
    def cropOutput(out2D, imageHeight, imageWidth, outputHeight, outputWidth):
        A = np.zeros((imageHeight, imageWidth), dtype=int)
        for i in range(imageHeight):
            for j in range(imageWidth):
                A[i][j] = out2D[i + (outputHeight - imageHeight)][j + outputWidth - imageWidth]

        return A

    @staticmethod
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

    def myConv2(self, A, k):
        windowHeight = len(k)
        windowWidth = len(k[0])
        paddedIm = self.paddImage(A, windowHeight, windowWidth)
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
        # return out2D
        return self.cropOutput(out2D, len(A), len(A[0]), outputHeight, outputWidth)

    def gaussian(self, A):
        height = len(A)
        width = len(A[0])
        for i in range(height):
            for j in range(width):
                A[i][j] += random.randint(-50, 50)

        #normalized = preprocessing.normalize(A)

        #min = np.min(A)
        #max = np.max(A)
        #A += abs(np.min(A))

        #for i in range(height):
        #    for j in range(width):
        #        A[i][j] = ((A[i][j] - min) / (max - min)) * 255.0
        #        A[i][j] = int(A[i][j])

        return A

    def saltAndPaper(self, A):
        for i in range(len(A)):
            for j in range(len(A[0])):
                if random.uniform(0.0, 1.0) < 0.5:  # random probability
                    if random.uniform(0.0, 1.0) < 0.5:
                        A[i][j] = 0
                    else:
                        A[i][j] = 254

        return A

    def myImNoise(self, A, param):
        A = np.array(A)
        B = np.array(A.copy())
        if param == "gaussian":
            self.processedImage = self.gaussian(B)
        else:
            self.processedImage = self.saltAndPaper(B)

    @staticmethod
    def findMedian(W):
        # w1D = np.reshape(W, (height * width, -1))
        w1Dsorted = np.sort(W)
        w1Dunique = np.unique(w1Dsorted)
        # print(w1Dunique)
        median = w1Dunique[len(w1Dunique) // 2]

        return median

    def median(self, A):
        windowHeight = 5
        windowWidth = 5
        paddedIm = self.paddImage(A, windowHeight, windowWidth)
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
                medianImage.append(self.findMedian(temp))

        medianImage = np.array(medianImage)
        out2D = np.reshape(medianImage, (outputHeight, -1))  # maybe output height second parameter
        # return out2D
        return self.cropOutput(out2D, len(A), len(A[0]), outputHeight, outputWidth)

    def myImFilter(self, A, param):
        if param == "mean":
            height = 16
            width = 16
            kernelF = np.full((height, width), (1 / (height * width)))
            kernelF = np.reshape(kernelF, (height, -1))
            # kernelF = np.array([[1.0 / 9, 1.0 / 9, 1.0 / 9],
            #                    [1.0 / 9, 1.0 / 9, 1.0 / 9],
            #                    [1.0 / 9, 1.0 / 9, 1.0 / 9]])
            outMean = self.myConv2(A, kernelF)
            self.processedImage = outMean
        else:
            self.processedImage = self.median(A)


