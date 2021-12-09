from tkinter import *
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog
import cv2

# documentation https://www.pyimagesearch.com/2016/05/23/opencv-with-tkinter/

def select_image():
    # grab a reference to the image panels
    global panelA, panelB

    # open a file chooser dialog and allow the user to select an input
    # image
    path = filedialog.askopenfilename()


select_image()
