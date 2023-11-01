from argparse import ArgumentParser
import os
from natsort import natsorted
import glob
import numpy as np
from PIL import Image
# from mmseg.core.evaluation import get_palette

from mmseg.apis import init_model, inference_model, show_result_pyplot
from mmengine import Config
import time

PALETTE = [ [148, 128, 106],   # 0 = Terrain
                [197, 185, 172],   # 1 = Unpaved Route
                [77, 69, 59],      # 2 = Paved Road
                [127, 122, 112],   # 3 = Tree Trunk
                [102, 127, 87],    # 4 = Tree Foliage
                [145, 145, 145],   # 7 = Rocks
                [158, 217, 92],    # 8 = Large Shrubs
                [200, 217, 92],    # 5 = Low Vegetation
                [216, 157, 92],    # 10 = Wire Fence
                [0, 0, 255],       # 9 = Sky
                [255, 0, 0],       # 11 = Person
                [53, 133, 193],    # 12 = Vehicle
                [0, 255, 0],       # 6 = Building
                [77, 69, 59],      # 2 = Paved Road
                [255, 0, 255],     # 13 = Misc
                [0, 185, 191],     # 14 = Water
                [65, 0, 74]   ,    # 15 = Animal
                [255, 255, 255],   # 16 =
                [255, 255, 255],   # 17 = Ignore
     ]

palette = np.array(PALETTE)
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

def save_segmentation(seg,path,img_name,palette=None,num_classes=19):

    Seg_dir = '/home/iariav/Deep/Pytorch/res/' + 'Seg_test_rgb'
    if not os.path.exists(Seg_dir):
        os.makedirs(Seg_dir)

    out_filename = os.path.join(Seg_dir , img_name)

    seg_logit = seg.seg_logits.data.cpu()
    seg_pred = seg_logit.argmax(dim=0)
    ids = np.unique(seg_pred)[::-1]
    legal_indices = ids < num_classes
    ids = ids[legal_indices]
    labels = np.array(ids, dtype=np.int64)

    # colors = [palette[label] for label in labels]

    seg_pred[seg_pred > 15] = 14
    seg_pred = np.squeeze(seg_pred)

    color_seg = np.zeros((seg_pred.shape[0], seg_pred.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[seg_pred == label, :] = color

    # # save the results
    im = Image.fromarray(color_seg)
    # im = im.resize((640,512))
    im.save(out_filename)

def save_depth(depth_pred,img_name,palette=None,num_classes=19):

    Depth_dir = '/home/iariav/Deep/Pytorch/res/' + 'Depth_test_rgb'
    if not os.path.exists(Depth_dir):
        os.makedirs(Depth_dir)

    out_filename = os.path.join(Depth_dir , img_name)

    depth_pred = np.array((depth_pred.cpu())*(255.0/65)).astype(np.uint8)
    depth_pred = np.squeeze(depth_pred)

    # depth_pred*=(255.0/65)
    # # save the results
    im = Image.fromarray(depth_pred)
    # im = im.resize((640, 512))
    im.save(out_filename)

def main():
    parser = ArgumentParser()
    parser.add_argument('path', help='Images folder path')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='cityscapes',
        help='Color palette used for segmentation map')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    images = list_images(args.path)

    for image_name in images:

        subdir_path = os.path.dirname(image_name)
        dirname = os.path.basename(args.path)
        dirlen = len(os.path.basename(args.path))
        maskfolder = os.path.join(args.path[:(-dirlen)], dirname + "_Mask_NN")
        img = os.path.join(args.path,image_name)
        # if not os.path.exists(maskfolder):
        #     os.makedirs(maskfolder)
        #     print("Created folder {}".format(maskfolder))

        start = time.time()
        result = inference_model(model, img)
        end = time.time() - start
        print('Classification took {} ms'.format(end * 1000))
        # save_segmentation(result, args.path, image_name, palette)
        save_depth(result.pred_depth.data, image_name, palette)


if __name__ == '__main__':
    main()
