import imageio
import glob
import os
image_folder = ''
video_name = 'video.mp4'
images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
writer = imageio.get_writer(video_name, fps=5)
for image in images:
    writer.append_data(imageio.imread(os.path.join(image_folder, image)))
writer.close()
