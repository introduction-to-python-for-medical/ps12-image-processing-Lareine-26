from PIL import Image
import numpy as np
from scipy.signal import convolve2d
from skimage.filters import median
from skimage.morphology import ball

def load_image(path):
    """
    Load an image and convert it to a numpy array.
    
    Parameters:
        path (str): Path to the image file.
        
    Returns:
        np.ndarray: The loaded image as a numpy array.
    """
    image = Image.open(path).convert('RGB')  # Ensure RGB format
    return np.array(image)

def edge_detection(image):
    """
    Perform edge detection on an image using Sobel filters.
    
    Parameters:
        image (np.ndarray): Input image array (RGB).
        
    Returns:
        np.ndarray: Magnitude of edges detected in the image.
    """
    # Convert to grayscale using correct weights
    gray_image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

    # Sobel kernels
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    # Apply convolution
    edge_x = convolve2d(gray_image, kernel_x, mode='same', boundary='symm')
    edge_y = convolve2d(gray_image, kernel_y, mode='same', boundary='symm')

    # Compute edge magnitude correctly
    edge_mag = np.sqrt(edge_x**2 + edge_y**2)  # Corrected **2 exponent
    return edge_mag
