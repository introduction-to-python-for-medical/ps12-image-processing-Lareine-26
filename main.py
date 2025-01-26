from skimage.filters import median
from skimage.morphology import ball
from image_utils import load_image, edge_detection
from PIL import Image
from skimage.filters import median
from skimage.morphology import ball
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
image_array = load_image('/content/lena.jpg')
clean_image = median(image_array, ball(3))
edgeMAG = edge_detection(clean_image)
plt.imshow(edgeMAG, cmap='gray')
plt.show()

plt.imshow(clean_image, cmap='gray')
plt.title("Cleaned Image")
plt.axis("off")
plt.show()

# Step 3: Apply edge detection
edgeMAG = edge_detection(clean_image)

# Display the edge-detected image
plt.imshow(edgeMAG, cmap='gray')
plt.title("Edge Detected Image")
plt.axis("off")
plt.show()

# Step 4: Visualize the histogram to determine a threshold value
image_1d = image_array.flatten() #converts to 1D
plt.figure(figsize=(5, 3))
plt.hist(image_1d, bins=255, range=(0, 255))
plt.show()


# Step 5: Convert to binary array
threshold_value = 100  # Adjust based on the histogram
edge_binary = edgeMAG > threshold_value

# Display the binary edge-detected image
plt.imshow(edge_binary, cmap='gray')
plt.title("Binary Edge Image")
plt.axis("off")
plt.show()

# Step 6: Save the binary edge-detected image
binary_image_uint8 = (edge_binary * 255).astype(np.uint8)
edge_image = Image.fromarray(binary_image_uint8)
edge_image.save("my_edges.png")
print("Binary edge image saved as 'my_edges.png'")

