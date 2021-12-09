from tkinter import *
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog
import cv2
import numpy

# documentation https://www.pyimagesearch.com/2016/05/23/opencv-with-tkinter/

# grab a reference to the image panels
#global panelA, panelB


def select_image():
    # grab a reference to the image panels
    global panelA, panelB

    # open a file chooser dialog and allow the user to select an input
    # image
    path = filedialog.askopenfilename()
    return path

# initialize the window toolkit along with the two image panels
root = Tk()
panelA = None
panelB = None

# create a button, then when pressed, will trigger a file chooser
# dialog and allow the user to select an input image; then add the
# button the GUI
btn = Button(root, text="Select an image", command=select_image)
btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")

# kick off the GUI
root.mainloop()

thePath = select_image()
img = None
dg = None

# ensure a file path was selected
if len(thePath) > 0:
    # load the image from disk, convert it to grayscale, and detect
    # edges in it
    image = cv2.imread(thePath, 0)  # read image - black and white    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # OpenCV represents images in BGR order; however PIL represents
    # images in RGB order, so we need to swap the channels
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # convert the images to PIL format...
    image = Image.fromarray(image)
    edged = Image.fromarray(numpy.zeros((image.height, image.width)))

    # ...and then to ImageTk format
    image = ImageTk.PhotoImage(image)
    edged = ImageTk.PhotoImage(edged)

    img = image
    dg = edged

# if the panels are None, initialize them
if panelA is None or panelB is None:
    # the first panel will store our original image
    panelA = Label(image=img)
    panelA.image = img
    panelA.pack(side="left", padx=10, pady=10)

    # while the second panel will store the edge map
    panelB = Label(image=dg)
    panelB.image = dg
    panelB.pack(side="right", padx=10, pady=10)

    # otherwise, update the image panels

    # update the pannels
    panelA.configure(image=img)
    panelB.configure(image=dg)
    panelA.image = dg
    panelB.image = dg

