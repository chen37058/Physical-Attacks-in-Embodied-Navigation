import imageio
import os

image_folder = 'bledner/viewpoints-camouflage'
video_name = 'docs/video.mp4'

# Get all files in the folder that end with .png and sort them by filename
images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")])

# Create a video writer object
writer = imageio.get_writer(video_name, fps=8)

# Iterate through the sorted list of images and write each image into the video
for image in images:
    writer.append_data(imageio.imread(os.path.join(image_folder, image)))

# Close the video writer
writer.close()