from tkinter import *
from tkinter.ttk import *
from tkinter import filedialog
from PIL import Image
from PIL import ImageTk
import cv2


def initialize():
    root = Tk()
    root.title("DIP app")
    root.geometry('1280x820')
    return root


def labels(rt):
    w = Label(rt, text='Apply filters')
    # w.pack()


def buttons(rt):

    b2 = Button(rt, text='Select Image', command=selectImage)
    b2.place(relx=0.5, rely=0.5, anchor=CENTER)

    return b2


def selectImage():
    path = filedialog.askopenfilename()
    if len(path) > 0:
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)

        b2.destroy()
        frameImage_Options(image)


def frameImage_Options(img):
    #canvas = Canvas(root, width=500, height=500)
    #canvas.pack()
    #canvas.create_image(100, 100, anchor=NW, image=img)

    bottomFrame = Frame(root)
    bottomFrame.pack(side=BOTTOM)

    b1 = Button(bottomFrame, text='Apply Filter', command=selectImage)
    b1.pack(side=BOTTOM, padx=10, pady=10)



root = initialize()
labels(root)
b2 = buttons(root)
root.mainloop()
