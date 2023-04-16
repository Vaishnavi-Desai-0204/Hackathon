import cv2
import numpy as np

# Load the JPEG images and resize them to 224x224
wardrobe_images = []
for i in range(2):
    img = cv2.imread(f'wardrobe/clothing{i+1}.jpg', cv2.IMREAD_COLOR)
    img_resized = cv2.resize(img, (224, 224))
    wardrobe_images.append(img_resized)

# Convert the images to NumPy arrays and save them in wardrobe.npy
wardrobe_array = np.array(wardrobe_images)
np.save('wardrobe.npy', wardrobe_array)

