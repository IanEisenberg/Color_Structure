
# convert from LAB to RGB space
from skimage.color import lab2rgb

def pixel_lab2rgb(lst):
    lst = [float(x) for x in lst]
    return lab2rgb([[(lst)]]).flatten()