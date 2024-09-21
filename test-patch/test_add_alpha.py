import numpy as np
from PIL import Image

# Function to add an alpha channel with 50% opacity to the image
def add_alpha_channel_with_opacity(image_path):
    # Open the image
    image = Image.open(image_path).convert("RGBA")  # Ensure the image has an RGBA mode
    
    # Split the image into R, G, B, and A channels
    r, g, b, _ = image.split()
    
    # Create an alpha channel with 50% opacity
    alpha = Image.new('L', image.size, int(255 * 0.5))  # 'L' mode for single channel (grayscale)
    
    # Merge R, G, B with the new alpha channel
    image_with_alpha = Image.merge('RGBA', (r, g, b, alpha))
    
    # Save the modified image, replacing the original file
    image_with_alpha.save(image_path)
    
    # Read the saved image and output its shape
    loaded_image = Image.open(image_path)
    print(f"Shape of image with alpha channel: {np.array(loaded_image).shape}")

import numpy as np
from PIL import Image

# Function to read a grayscale image as an alpha channel, enhance RGB values, and save the image
def enhance_rgb_with_alpha_channel(image_path, alpha_path, output_path):
    # Open the RGB image
    rgb_image = Image.open(image_path).convert("RGB")
    
    # Open the grayscale image to be used as alpha channel
    alpha_image = Image.open(alpha_path).convert("L")
    
    # Convert images to numpy arrays
    rgb_array = np.array(rgb_image)
    alpha_array = np.array(alpha_image)
    
    # Enhance RGB values by 2 (ensure values are capped at 255)
    enhanced_rgb_array = np.clip(rgb_array, 0, 255).astype(np.uint8)
    
    # Create an RGBA image using the enhanced RGB values and the alpha channel
    rgba_array = np.dstack((enhanced_rgb_array, alpha_array))
    
    # Convert the numpy array to a PIL image
    enhanced_image = Image.fromarray(rgba_array, 'RGBA')
    
    # Save the enhanced image
    enhanced_image.save(output_path)
    
    # Read the saved image and output its shape
    loaded_image = Image.open(output_path)
    print(f"Shape of enhanced image with alpha channel: {np.array(loaded_image).shape}")

# Paths to the images
image_path = 'adversarial_patch.png'
alpha_path = 'adversarial_patch_mask.png'
output_path = 'enhanced_image_with_alpha.png'

# Enhance RGB with alpha channel and print the shape
# enhance_rgb_with_alpha_channel(image_path, alpha_path, output_path)

# Add alpha channel and print the shape
add_alpha_channel_with_opacity(image_path)
