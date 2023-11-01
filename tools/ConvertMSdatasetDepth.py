from argparse import ArgumentParser
import os
import os.path as osp
from natsort import natsorted
import glob
import numpy as np
from PIL import Image
from imageio import imread
from scipy.io import loadmat
# from mmseg.core.evaluation import get_palette
import mmengine
from mmseg.apis import init_model, inference_model, show_result_pyplot
from mmengine import Config
import time

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

# Raw thermal radiation value to tempearture


def load_as_float_depth(path):
    if 'png' in path:
        depth =  np.array(imread(path).astype(np.float32))
    elif 'npy' in path:
        depth =  np.load(path).astype(np.float32)
    elif 'mat' in path:
        depth =  loadmat(path).astype(np.float32)
    return depth
def main():
    parser = ArgumentParser()
    parser.add_argument('path', help='Images folder path')
    args = parser.parse_args()

    folders = [f.path for f in os.scandir(args.path) if f.is_dir()]

    for i,folder in enumerate(folders):
        path = folder + '/'
        print('Started working on folder {}:{}'.format(i,folder))
        images = list_images(path,pattern='thr/depth_filtered/*')
        print('Found {} images.'.format(len(images)))

        for image_name in images:

            img_name = os.path.join(path,image_name)
            try:
                depth = load_as_float_depth(img_name) / 256.0
            except Exception:
                continue

            image_split = img_name.split('/')
            dst_dir = os.path.join('/hdd_data/MSdataset', image_split[-4], image_split[-3], image_split[-2])
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)
            dst_name = os.path.join(dst_dir, image_split[-1].replace('png','npz'))

            np.savez_compressed(dst_name,depth)

        print('Finished converting.')

if __name__ == '__main__':
    main()
