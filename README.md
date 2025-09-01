# Install OpenCV if not already installed
!pip install opencv-python matplotlib

import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

# Step 1: Upload an image
uploaded = files.upload()

# Get uploaded file name
image_path = list(uploaded.keys())[0]

# Read the original image
original_img = cv2.imread(image_path)

# Convert BGR (OpenCV default) to RGB for display in matplotlib
original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

# Step 2: Resize to 224x224
resized_img = cv2.resize(original_img_rgb, (224, 224))

# Step 3: Rescale pixel values to [0,1]
rescaled_img = resized_img.astype(np.float32) / 255.0

# Step 4: Convert to Grayscale
gray_img = cv2.cvtColor(resized_img, cv2.COLOR_RGB2GRAY)

# Step 5: Apply Canny edge detection
edges = cv2.Canny(gray_img, threshold1=100, threshold2=200)

# Display all images
plt.figure(figsize=(15, 8))

# Original
plt.subplot(1, 5, 1)
plt.imshow(original_img_rgb)
plt.title("Original Image")
plt.axis("off")

# Resized
plt.subplot(1, 5, 2)
plt.imshow(resized_img)
plt.title("Resized (224x224)")
plt.axis("off")

# Rescaled
plt.subplot(1, 5, 3)
plt.imshow(rescaled_img)
plt.title("Rescaled [0,1]")
plt.axis("off")

# Grayscale
plt.subplot(1, 5, 4)
plt.imshow(gray_img, cmap="gray")
plt.title("Grayscale")
plt.axis("off")

# Edge Detection
plt.subplot(1, 5, 5)
plt.imshow(edges, cmap="gray")
plt.title("Canny Edges")
plt.axis("off")

plt.show()
