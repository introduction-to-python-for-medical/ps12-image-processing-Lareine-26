from PIL import Image
import numpy as np
from scipy.signal import convolve2d
from skimage.filters import median
from skimage.morphology import ball

def load_image(path):
    """
    Load an image and convert it to a grayscale numpy array.
    
    Parameters:
        path (str): Path to the image file.
        
    Returns:
        np.ndarray: The loaded image as a grayscale numpy array.
    """
    image = Image.open(path).convert('L')  # Convert directly to grayscale
    return np.array(image)

def edge_detection(image):
    """
    Perform edge detection on an image using Sobel filters.
    
    Parameters:
        image (np.ndarray): Input grayscale image.
        
    Returns:
        np.ndarray: Magnitude of edges detected in the image.
    """
    # Sobel kernels
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    # Apply convolution
    edge_x = convolve2d(image, kernel_x, mode='same', boundary='symm')
    edge_y = convolve2d(image, kernel_y, mode='same', boundary='symm')

    # Compute edge magnitude correctly
    edge_mag = np.sqrt(edge_x**2 + edge_y**2)
    return edge_mag.astype(np.uint8)  # Ensure uint8 output
