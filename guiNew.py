from tkinter import *
from tkinter.ttk import *
from tkinter import filedialog
from PIL import Image
from PIL import ImageTk
import cv2

class Data:
    def __init__(self):
        self.initialImage = None
        self.filteredImage = None
        self.noisyImage = None

    def addImage(self, img):
        """array style"""
        self.initialImage = img



def initialize():
    root = Tk()
    root.title("DIP app")
    root.geometry('1280x820')
    return root


def labels(rt):
    w = Label(rt, text='Apply filters', font=("Arial", 25))
    w.pack()


def buttons(rt):
    b2 = Button(rt, text='Select Image', command=selectImage)
    b2.place(relx=0.5, rely=0.5, anchor=CENTER)

    return b2


def selectImage():
    path = filedialog.askopenfilename()
    if len(path) > 0:
        image = cv2.imread(path)
        # computations for resizing
        height = len(image)
        width = len(image[0])
        a = height / 700
        width = int(width / a)
        height = 700
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = image.resize((width, height), Image.ANTIALIAS)
        image = ImageTk.PhotoImage(image)

        frameImage_Options(image, height, width)



def frameImage_Options(img, h, w):
    canvas = Canvas(root, width=w, height=h)
    canvas.create_image(10, 10, anchor=NW, image=img)
    canvas.place(relx=0.5, rely=0.5, anchor=CENTER)

    bottomFrame = Frame(root)
    bottomFrame.pack(side=BOTTOM)

    b1 = Button(bottomFrame, text='Apply Filter', command=None)
    b1.pack(side=BOTTOM, padx=10, pady=10)

    canvas.mainloop()

root = initialize()
labels(root)
b2 = buttons(root)
root.mainloop()
