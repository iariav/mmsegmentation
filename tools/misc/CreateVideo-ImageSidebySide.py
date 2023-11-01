import numpy as np
import os
import cv2
import glob
from tqdm import tqdm
from natsort import natsorted
from PIL import Image

def list_images(folder, pattern='*', ext='png'):
#def list_images(folder, pattern='Day*', ext='png'):
    """List the images in a specified folder by pattern and extension

    Args:
        folder (str): folder containing the images to list
        pattern (str, optional): a bash-like pattern of the files to select
                                 defaults to * (everything)
        ext(str, optional): the image extension (defaults to png)

    Returns:
        str list: list of (filenames) images matching the pattern in the folder
    """
    os.chdir(folder)
    filenames = natsorted(glob.glob('**/'+ pattern + '.' + ext, recursive = True), key=lambda y: y.lower())
    return filenames

images_path = '/ssd_data/NAS/thermal/ROS/Ein_hashofet/2023-05-09/ein_hashofet4_0/ThermalImages'
masks_path = '/home/iariav/Deep/Pytorch/res/Seg_fp16'
depths_path = '/home/iariav/Deep/Pytorch/res/Depth_fp16'

images = list_images(images_path)
depths = list_images(depths_path)
masks = list_images(masks_path)
video_name = '/home/iariav/Deep/Pytorch/mmsegmentation/work_dirs/b1_sgd_pixelformer_thermal_win7/res_trt_fp16.avi'

images = natsorted(images, key=lambda y: y.lower())
depths = natsorted(depths, key=lambda y: y.lower())
masks = natsorted(masks, key=lambda y: y.lower())

img = Image.open(images[0])
shape = img.size

vid_height = shape[1]
vid_width = shape[0]*3

fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter(video_name, fourcc, 2, (vid_width,vid_height))

for i in tqdm(range(len(images))):
    a= 0+i
    if a == range(len(images)):
        break

    image = cv2.imread(os.path.join(images_path, images[a]))
    mask = cv2.imread(os.path.join(masks_path, masks[a]))
    depth = cv2.imread(os.path.join(depths_path, depths[a]))

    numpy_horizontal_concat = np.concatenate((mask,image,depth), axis=1)
    video.write(numpy_horizontal_concat)

video.release()


