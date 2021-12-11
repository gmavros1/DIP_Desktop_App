from tkinter import *
from tkinter.ttk import *
from tkinter import filedialog
from PIL import Image
from PIL import ImageTk
import cv2
from app import Data


class Gui:
    def __init__(self):
        self.windowWidth = 1280
        self.windowHeight = 820
        self.root = Tk()
        self.root.title("DIP app")
        windowSize = str(self.windowWidth) + "x" + str(self.windowHeight)
        self.root.geometry(windowSize)
        # App data
        self.data = Data()
        # Buttons
        self.btn1 = None  # Select Image
        self.btn2 = None  # Apply Filter
        self.btn3 = None  # Add Noise

        # Canvas for image display
        self.canvas = None

        # Run
        self.run()

    def selectImage(self):  # Image --> openCV style
        path = filedialog.askopenfilename()
        if len(path) > 0:
            image = cv2.imread(path)
            self.data.addImage(image)
            self.btn1.destroy()
            self.otherButtons()
            self.displayImage(self.data.initialImage)

    def selectImageButton(self):
        self.btn1 = Button(self.root, text='Select Image', command=self.selectImage)
        self.btn1.place(relx=0.5, rely=0.5, anchor=CENTER)

    def otherButtons(self):
        self.btn2 = Button(self.root, text='Apply Filter', command=None)
        #self.btn2.pack(side=BOTTOM, padx=10, pady=10)

        self.btn3 = Button(self.root, text='Add Noise', command=None)
        #self.btn3.pack(side=BOTTOM, padx=10, pady=10)

        self.btn2.grid(row=0, column=0, sticky=W, pady=2)
        self.btn3.grid(row=0, column=1, sticky=W, pady=2)

    def displayImage(self, img):
        image = img
        height = len(image)
        width = len(image[0])
        a = height / 700
        width = int(width / a)
        height = 700
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = image.resize((width, height), Image.ANTIALIAS)
        image = ImageTk.PhotoImage(image)

        self.canvas = Canvas(self.root, width=width, height=height)
        self.canvas.create_image(10, 10, anchor=NW, image=image)
        self.canvas.place(relx=0.5, rely=0.5, anchor=CENTER)
        self.canvas.mainloop()

    def run(self):
        self.selectImageButton()
        self.root.mainloop()


gui = Gui()
