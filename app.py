class App:
    def __init__(self):
        self.initialImage = None
        self.filteredImage = None
        self.noisyImage = None

    def addImage(self, img):
        """array style"""
        self.initialImage = img
