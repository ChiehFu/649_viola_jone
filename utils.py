import numpy as np

class RectangleRegion:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    def compute_feature(self, ii):
        return ii[self.y+self.height - 1][self.x+self.width - 1] + ii[self.y - 1][self.x - 1] - (ii[self.y+self.height - 1][self.x - 1] + ii[self.y - 1][self.x+self.width - 1])
    

def integral_image(image):
    ii = np.zeros(image.shape)
    s = np.zeros(image.shape)
    for y in range(len(image)):
        for x in range(len(image[y])):
            s[y][x] = s[y-1][x] + image[y][x] if y-1 >= 0 else image[y][x]
            ii[y][x] = ii[y][x-1]+s[y][x] if x-1 >= 0 else s[y][x]
    return ii