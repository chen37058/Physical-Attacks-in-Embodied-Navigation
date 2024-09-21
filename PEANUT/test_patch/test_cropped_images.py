import os
from PIL import Image

# Define directory path and cropping area
input_dir = ''
output_dir = 'cropped_images'
crop_x = 100
crop_y = 50
crop_width = 480
crop_height = 480

# Create output directory (if it doesn't exist)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Iterate through all image files in the directory
for filename in os.listdir(input_dir):
    if filename.lower().endswith('.jpg'):
        filepath = os.path.join(input_dir, filename)
        # Open the image
        with Image.open(filepath) as img:
            # Crop the image
            cropped_img = img.crop((crop_x, crop_y, crop_x + crop_width, crop_y + crop_height))
            # Save the cropped image
            cropped_img.save(os.path.join(output_dir, filename))

print("Image cropping completed!")
