from image_utils import load_image, edge_detection
from PIL import Image
import numpy as np
from skimage.filters import median
from skimage.morphology import ball

path='lena.jpg'
image=load_image(path)
clean_image = median(image, ball(3))
edge_mag=edge_detection(clean_image)
edge_binary=edge_mag<100
edge_image = Image.fromarray((edge_binary*255).astype(np.uint8))
edge_image.save('my_edges.png')
