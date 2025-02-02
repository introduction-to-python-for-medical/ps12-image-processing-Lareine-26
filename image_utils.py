from PIL import Image
import numpy as np
from scipy.signal import convolve2d

def load_image(file_path):
    image = Image.open(file_path)
    image = np.array(image)
    return image
def edge_detection(image):
    gray_image = np.mean(image, axis=2)
    kernelY = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])
    kernelX = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    edge_x=convolve2d(gray_image, kernelX, mode='same', boundary='symm')
    edge_y=convolve2d(gray_image, kernelY, mode='same', boundary='symm')
    edge_mag=np.sqrt(edge_x**2+edge_y**2)
    return edge_mag
