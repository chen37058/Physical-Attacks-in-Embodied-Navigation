import numpy as np
from PIL import Image

# Function to generate a random Gaussian noise image
def generate_gaussian_noise_image(width, height):
    # Create a random Gaussian noise array with mean 0 and standard deviation 1
    noise = np.random.normal(0, 1, (height, width, 3))
    
    # Normalize the noise to the 0-255 range for image format
    noise = (255 * (noise - noise.min()) / (noise.max() - noise.min())).astype(np.uint8)
    
    # Convert the noise array to an image
    noise_image = Image.fromarray(noise)
    # Save the image as PNG
    noise_image.save('adversarial_patch.png')
    
    # Read the saved image and output its shape
    loaded_noise_image = Image.open('adversarial_patch.png')
    print(f"Shape of Gaussian noise image: {np.array(loaded_noise_image).shape}")

# Function to generate an image with black color and 60% opacity
def generate_black_opacity_image(width, height):
    # Set opacity value，The lower the more transparent
    opacity_value = int(255 * 0.6)

    # Create a fully opaque black mask
    opacity_mask = np.full((height, width), opacity_value, dtype=np.uint8)

    # Convert NumPy array to PIL image
    opacity_image = Image.fromarray(opacity_mask, 'L')  # 'L' indicates single-channel grayscale image

    # Save image as PNG file
    opacity_image.save('adversarial_patch_mask.png')
    
    # Read the saved image and output its shape
    loaded_opacity_image = Image.open('adversarial_patch_mask.png')
    print(f"Shape of black opacity image: {np.array(loaded_opacity_image).shape}")

# Function to generate an image with random opacity values
def generate_random_opacity_image(width, height):
    # Create a random opacity mask with values between 0 and 255
    opacity_mask = np.random.randint(0, 256, (height, width), dtype=np.uint8)

    # Convert NumPy array to PIL image
    opacity_image = Image.fromarray(opacity_mask, 'L')  # 'L' indicates single-channel grayscale image

    # Save image as PNG file
    opacity_image.save('adversarial_patch_random_opacity.png')
    
    # Read the saved image and output its shape
    loaded_opacity_image = Image.open('adversarial_patch_random_opacity.png')
    print(f"Shape of random opacity image: {np.array(loaded_opacity_image).shape}")

# Set the image size
width, height = 512, 512

# Generate and save the images
generate_gaussian_noise_image(width, height)
generate_black_opacity_image(width, height)
# generate_random_opacity_image(width, height)
