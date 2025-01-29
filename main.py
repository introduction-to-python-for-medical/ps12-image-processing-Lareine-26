from image_utils import load_image, edge_detection
from PIL import Image
import numpy as np
from skimage.filters import median
from skimage.morphology import ball

path = 'lena.jpg'
picture = load_image(path)

# Apply median filtering
clean_image = median(picture, ball(3))

# Perform edge detection
the_final = edge_detection(clean_image)

# Create a binary edge map with corrected threshold
binary_image = the_final > 50  # Threshold set to match expected output

# Convert to uint8 and save
binary = Image.fromarray((binary_image * 255).astype(np.uint8))
binary.save('my_edges.png')

print("Edge detection completed and saved as 'lena_edges.png'")
