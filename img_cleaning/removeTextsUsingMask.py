import os
import cv2
import numpy as np
from PIL import Image

# Define paths
input_dir = "raw_imgs/haemorrhagic"
output_dir = "cleaned_imgs/haemorrhagic"
mask_path = "img_cleaning/mask/mask.png"

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the mask
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
if mask is None:
    print("Error: Could not load mask.png")
    exit()

# Process each JPG file in the input directory
for filename in os.listdir(input_dir):
    if filename.lower().endswith('.jpg'):
        # Construct full file paths
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        # Load the original image
        image = cv2.imread(input_path)
        if image is None:
            print(f"Warning: Could not load {filename}")
            continue
            
        # Ensure mask and image have the same dimensions
        if image.shape[:2] != mask.shape[:2]:
            # Resize mask to match image dimensions
            mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
        else:
            mask_resized = mask
            
        # Apply the mask directly
        masked_image = cv2.bitwise_and(image, image, mask=mask_resized)
        
        # Save the result
        success = cv2.imwrite(output_path, masked_image)
        if success:
            print(f"Processed and saved: {filename}")
        else:
            print(f"Error saving: {filename}")

print("Processing complete!")