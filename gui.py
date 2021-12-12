from tkinter import *
from tkinter.ttk import *
from tkinter import filedialog
from PIL import Image
from PIL import ImageTk
import cv2
from app import App
import numpy as np


class Gui:
    def __init__(self):
        self.windowWidth = 1280
        self.windowHeight = 820
        self.root = Tk()
        self.root.title("DIP app")
        # windowSize = str(self.windowWidth) + "x" + str(self.windowHeight)
        windowSize = str("200x300")
        self.root.geometry(windowSize)
        self.root.eval('tk::PlaceWindow . center')
        # App storing and data processing
        self.data = App()

        # Buttons
        self.btn1 = None  # Select Image
        self.btn2 = None  # Apply Filter
        self.btn3 = None  # Add Noise
        self.btn4 = None  # Show First Image
        self.otherButtonFrame = None

        # Canvas for image display
        self.canvas = None

        # Run
        self.run()

    def selectImage(self):  # Image --> openCV style
        path = filedialog.askopenfilename()
        if len(path) > 0:
            image = cv2.imread(path, 0)
            self.data.addImage(image)
            self.btn1.destroy()
            self.otherButtons()
            self.displayImage(self.data.initialImage)

    def selectImageButton(self):
        self.btn1 = Button(self.root, text='Select Image', command=self.selectImage)
        self.btn1.place(relx=0.5, rely=0.5, anchor=CENTER)

    def addNoise(self):
        top = Toplevel(self.root)
        top.geometry("300x65")
        top.title("Noise")

        # Position pop up
        root_x = self.root.winfo_rootx()
        root_y = self.root.winfo_rooty()
        win_x = root_x + 300
        win_y = root_y + 100
        top.geometry(f'+{win_x}+{win_y}')

        # padOrSame = IntVar()  # 0 --> pad 1 --> same
        # padOrSame.set(0)
        # pad1 = Radiobutton(top, text="Original size", variable=padOrSame, value=1, command=None)
        # pad0 = Radiobutton(top, text="Padded size", variable=padOrSame, value=0, command=None)
        # pad0.grid(row=0, column=1, sticky=W, pady=4)
        # pad1.grid(row=1, column=1, sticky=W, pady=4)

        def gauss():
            self.data.myImNoise(self.data.processedImage, "gaussian")
            self.displayImage(self.data.processedImage)

        def salt():
            self.data.myImNoise(self.data.processedImage, "salt")
            self.displayImage(self.data.processedImage)

        gauss = Button(top, text='Gaussian', command=gauss)
        salt = Button(top, text='Salt', command=salt)

        gauss.grid(row=0, column=0, sticky=W, pady=4)
        salt.grid(row=1, column=0, sticky=W, pady=4)

    def applyFilter(self):
        top = Toplevel(self.root)
        top.geometry("300x65")
        top.title("Filters")

        # Position pop up
        root_x = self.root.winfo_rootx()
        root_y = self.root.winfo_rooty()
        win_x = root_x + 300
        win_y = root_y + 100
        top.geometry(f'+{win_x}+{win_y}')

        def mean():
            pass

        def median():
            self.data.myImFilter(self.data.processedImage, "median")
            self.displayImage(self.data.processedImage)

        mean = Button(top, text='Mean Filter', command=mean)
        median = Button(top, text='Median Filter', command=median)

        mean.grid(row=0, column=0, sticky=W, pady=4)
        median.grid(row=1, column=0, sticky=W, pady=4)

    def otherButtons(self):
        self.otherButtonFrame = Frame(self.root)
        self.otherButtonFrame.pack(side=BOTTOM)

        self.btn2 = Button(self.otherButtonFrame, text='Apply Filter', command=self.applyFilter)
        self.btn3 = Button(self.otherButtonFrame, text='Add Noise', command=self.addNoise)
        self.btn4 = Button(self.otherButtonFrame, text='Initial', command=self.displayInitial)

        self.btn2.grid(row=0, column=0, sticky=W, pady=2)
        self.btn3.grid(row=0, column=1, sticky=W, pady=2)
        self.btn4.grid(row=0, column=2, sticky=W, pady=2)

    def displayInitial(self):
        self.displayImage(self.data.initialImage)

    def displayImage(self, img):
        image = img
        height = len(image)
        width = len(image[0])
        a = height / 700
        width = int(width / a)
        height = 700
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray((image % 256).astype(np.uint8))
        image = image.resize((width, height), Image.ANTIALIAS)
        image = ImageTk.PhotoImage(image)

        sz = str(width) + "x" + str(self.windowHeight)
        self.root.geometry(sz)

        self.canvas = Canvas(self.root, width=width, height=height)
        self.canvas.create_image(10, 10, anchor=NW, image=image)
        self.canvas.place(relx=0.5, rely=0.5, anchor=CENTER)
        self.canvas.mainloop()

    def run(self):
        self.selectImageButton()
        self.root.mainloop()


gui = Gui()
