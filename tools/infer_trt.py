import numpy as np

from PIL import Image
import os
from natsort import natsorted
import glob
import time
# from mmseg.models.utils.wrappers import resize\
from skimage.transform import resize
# import onnxruntime

from polygraphy.backend.onnxrt import OnnxrtRunner, SessionFromOnnx
from polygraphy.backend.common import BytesFromPath
from polygraphy.backend.trt import TrtRunner, EngineFromNetwork, NetworkFromOnnxPath, SaveEngine, TrtRunner, EngineFromBytes, CreateConfig, Calibrator, Profile
from polygraphy.comparator import Comparator, DataLoader
from polygraphy.logger import G_LOGGER

mode = 'fp32'
trt_path = '/home/iariav/Deep/Pytorch/mmsegmentation/work_dirs/b1_sgd_pixelformer_thermal_win7/trt-poly-' + mode + '.engine'
images_path = '/ssd_data/NAS/thermal/ROS/Ein_hashofet/2023-05-09/ein_hashofet4_0/ThermalImages'
model_path = '/home/iariav/Deep/Pytorch/mmsegmentation/work_dirs/b1_sgd_pixelformer_thermal_win7/iter_360000_opset15_seg_predict.onnx'

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

mean=[123.675, 116.28, 103.53]
std=[58.395, 57.12, 57.375]
mean = np.array(mean, dtype=np.float32)
std = np.array(std, dtype=np.float32)
pallete = np.array(PALETTE)
load_trt_model = False
use_dynamic_shapes = False

mean_t=[122.5]
std_t=[57.375]
mean_t = np.array(mean_t, dtype=np.float32)
std_t = np.array(std_t, dtype=np.float32)

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
    #filenames = natsorted(os.listdir(folder))
    os.chdir(folder)
    filenames = natsorted(glob.glob('**/'+ pattern + '.' + ext, recursive = True), key=lambda y: y.lower())
    #filenames = natsorted(glob.glob(folder + pattern + '.' + ext), key=lambda y: y.lower())
    #filenames = sorted(glob.glob(folder + pattern + '.' + ext))
    print(filenames)
    return filenames

def load_image(image_name):
    img = np.array(Image.open(os.path.join(images_path, image_name)).resize((480, 320)))
    img = (img - mean) / std

    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)

    return img

def load_thermal_image(image_name):
    img = np.array(Image.open(os.path.join(images_path, image_name)).resize((640, 512)))
    img = (img - mean_t) / std_t

    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=0)

    return img


def save_segmentation(seg,path,img_name,pallete=None):

    Seg_dir = '/home/iariav/Deep/Pytorch/res/' + 'Seg_' + mode
    if not os.path.exists(Seg_dir):
        os.makedirs(Seg_dir)

    out_filename = os.path.join(Seg_dir , img_name)
    seg[seg>15]=14
    seg = np.squeeze(seg)

    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(pallete):
        color_seg[seg == label, :] = color

    # # save the results
    im = Image.fromarray(color_seg)
    # im = im.resize((640,512))
    im.save(out_filename)

def save_depth(depth_pred,img_name,palette=None,num_classes=19):

    Depth_dir = '/home/iariav/Deep/Pytorch/res/' + 'Depth_' + mode
    if not os.path.exists(Depth_dir):
        os.makedirs(Depth_dir)

    out_filename = os.path.join(Depth_dir , img_name)

    depth_pred = (depth_pred*(255.0/65)).astype(np.uint8)
    depth_pred = np.squeeze(depth_pred)

    # depth_pred*=(255.0/65)
    # # save the results
    im = Image.fromarray(depth_pred)
    # im = im.resize((640, 512))
    im.save(out_filename)

if load_trt_model:
    build_engine = EngineFromBytes(BytesFromPath(trt_path))
else:
    if mode == 'fp32':
        build_engine = EngineFromNetwork(NetworkFromOnnxPath(model_path),
                                                 config=CreateConfig(fp16=False,
                                                                     # builder_optimization_level=5,
                                                                     # # profiles=profiles,
                                                                     # use_dla=True
                                                                     ))
    else: # fp16
        build_engine = EngineFromNetwork(NetworkFromOnnxPath(model_path),
                                         config=CreateConfig(fp16=True,
                                                             # builder_optimization_level=5,
                                                             # # profiles=profiles,
                                                             # use_dla=True
                                                             ))

    build_engine = SaveEngine(build_engine, path=trt_path)

images = list_images(images_path)

with G_LOGGER.verbosity(G_LOGGER.VERBOSE), TrtRunner(build_engine) as runner:
    for image_name in images:

        # img = load_image(image_name)
        img = load_thermal_image(image_name)

        # NOTE: The runner owns the output buffers and is free to reuse them between `infer()` calls.
        # Thus, if you want to store results from multiple inferences, you should use `copy.deepcopy()`.
        start = time.time()
        outputs = runner.infer(feed_dict={"input": img})
        pred_depth = outputs['output_depth']
        pred_seg = outputs['output_seg']
        # pred_seg = np.squeeze(np.argmax(pred_seg, axis=1))
        end = time.time() - start
        print('TRT:: Classification took {} ms'.format(end * 1000))
        save_segmentation(pred_seg,images_path,image_name,pallete)
        save_depth(pred_depth, image_name)
        # print("Inference succeeded!")