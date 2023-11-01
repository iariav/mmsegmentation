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
def Raw2Celsius(Raw):
    R = 380747
    B = 1428
    F = 1
    O = -88.539
    Celsius = B / np.log(R / (Raw - O) + F) - 273.15;
    return Celsius

def load_as_float_img(path):
    img =  imread(path).astype(np.float32)
    if len(img.shape) == 2: # for NIR and thermal images
        img = np.expand_dims(img, axis=2)
    return img

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

    # build the model from a config file and a checkpoint file

    folders = ['_2021-08-13-16-31-10','_2021-08-13-16-50-57','_2021-08-13-17-06-04','_2021-08-13-21-18-04',
               '_2021-08-13-21-36-10','_2021-08-13-21-58-13','_2021-08-13-22-03-03','_2021-08-13-22-16-02',
               '_2021-08-13-22-36-41']

    folders = [f.path for f in os.scandir(args.path) if f.is_dir()]

    for folder in folders:
        path = folder + '/'
        print('Started working on folder {}'.format(folder))
        images = list_images(path,pattern='thr/depth/*')
        print('Found {} images.'.format(len(images)))
        psum = 0.0
        psum_sq = 0.0

        # max_val = 0
        # min_val = 255

        img_name = os.path.join(path, images[0])
        raw_img = load_as_float_img(img_name)
        image_size = raw_img.shape

        for image_name in images:

            img_name = os.path.join(path,image_name)
            try:
                raw_img = load_as_float_img(img_name)
            except Exception:
                continue

            tgt_depth_gt = load_as_float_depth(sample['tgt_depth_gt']) / 256.0
            temp_img = Raw2Celsius(raw_img)

            psum += np.sum(temp_img)
            psum_sq += np.sum(np.square(temp_img))

        # pixel count
        count = len(images) * image_size[0] * image_size[1]

        # mean and STD
        total_mean = psum / count
        total_var = (psum_sq / count) - (total_mean ** 2)
        total_std = np.sqrt(total_var)

        s1 = total_std.item()
        s2 = 57.375
        m1 = total_mean.item()
        m2 = 122.5

        print('Training data stats:')
        print('- mean: {:.4f}'.format(total_mean.item()))
        print('- std:  {:.4f}'.format(total_std.item()))

        # Convert images according to new mean and std
        for image_name in images:

            img_name = os.path.join(path,image_name)
            image_split = img_name.split('/')
            dst_dir = os.path.join('/hdd_data/MSdataset', image_split[-4], image_split[-3], image_split[-2])
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)
            dst_name = os.path.join(dst_dir,image_split[-1])

            try:
                raw_img = load_as_float_img(img_name)
            except Exception:
                continue

            temp_img = Raw2Celsius(raw_img)

            converted_img = temp_img - m1
            converted_img *= (s2/s1)
            converted_img += m2
            converted_img = np.clip(converted_img, 0, 255).astype(np.uint8)
            im = Image.fromarray(np.squeeze(converted_img))
            im.save(dst_name)

        print('Finished converting.')

if __name__ == '__main__':
    main()
