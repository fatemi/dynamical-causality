import os
import re
import cv2
import click
import numpy as np
from PIL import Image


atoi = lambda text: int(text) if text.isdigit() else text
natural_keys = lambda text: [ atoi(c) for c in re.split(r'(\d+)', text) ]


def concat(image_folder):
    subfolders = ['game', 'mu', 'gqtl', 'mugq']
    images = dict()
    for sf in subfolders:
        images[sf] = [img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith("png")]
        images[sf].sort(key=natural_keys)



def make_gif(image_folder, video_name):
    frames = [img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith("png")]
    frames.sort(key=natural_keys)
    frames = [ Image.open(os.path.join(image_folder, image)) for image in frames ]
    width, height = frames[0].size
    for _ in range(5):
        frames.append(frames[-1])
    for _ in range(3):
        frames.append(Image.fromarray( 0*np.ones((width, height), dtype=np.int8) ))
    frame_one = frames[0]
    frame_one.save(video_name, format="GIF", append_images=frames, save_all=True, duration=200, loop=5)


def generate_video(image_folder, video_name):	
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith("png")]
    images.sort(key=natural_keys)

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), 10, (width, height))
    
    # Appending the images to the video one by one
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))
    
    cv2.destroyAllWindows()  # Deallocating memories taken for window creation
    video.release()          # releasing the video generated


"""
How to run:

python video.py -p .\results\pong2\figs\obs -n obs.gif -g

or

python video.py -p .\results\pong2\figs\obs -n obs.avi
"""


@click.command()
@click.option('--path', '-p', help="path to images")
@click.option('--video_name', '-n', help="name of the video")
@click.option('--gif', '-g', is_flag=True, help="make gif")
def run(path, video_name, gif):
    if gif:
        make_gif(image_folder=path, video_name=video_name)
    else:
        generate_video(image_folder=path, video_name=video_name)


if __name__ == '__main__':
    run()