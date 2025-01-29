from image_utils import load_image, edge_detection
from PIL import Image
import numpy as np
from skimage.filters import median
from skimage.morphology import ball

path = 'lena.jpg'
picture = load_image(path)

# Apply median filtering with a ball-shaped structuring element
clean_image = median(picture, ball(3))

# Perform edge detection
the_final = edge_detection(clean_image)

# Create a binary image by thresholding correctly
my_edges = the_final > 50  # Fixed threshold matching the test case

# Convert the binary image to a PIL Image and save
binary = Image.fromarray((my_edges * 255).astype(np.uint8))  # Scale to 0-255 for saving
binary.save('my_edges.png')

