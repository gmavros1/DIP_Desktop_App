# demo1.py dummy script
# additional user functions can be added

# Iason Karakostas
# AIIA Lab - CSD AUTH
# Digital Image Processing
# Assignment 2021

import numpy as np
import cv2


def myConv2(A, B, param):
    # do 2 dimensional convolution here
    # param can be 'pad' or 'same'
    print('Hello')


def myImNoise(A, param):
    # add noise according to the parameters
    # param must be at least 'gaussian' and 'saltandpepper'
    print('Hello')


def myImFilter(A, param):
    # fitler image A according to the parameters
    # param must be at least 'mean' and 'median'
    print('Hello')


A = cv2.imread('test.jpg', 0)  # read image - black and white
cv2.imshow('image', A)  # show image
cv2.waitKey(0)  # wait for key press
cv2.destroyAllWindows()  # close image window

# B = myImNoise(A, 'gaussian')
# or
# B = myImNoise(A, 'saltandpepper')
# cv2.imshow('image',A) #show image
# cv2.waitKey(0) #wait for key press
# cv2.destroyAllWindows() #close image window

# C = myImFilter(B, 'mean')
# or
# C = myImFilter(B, 'median')
# cv2.imshow('image',A) #show image
# cv2.waitKey(0) #wait for key press
# cv2.destroyAllWindows() #close image window