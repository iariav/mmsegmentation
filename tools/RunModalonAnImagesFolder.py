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


####   Materials     ####


# PALETTE = [ [148, 128, 106],   # 0 = Terrain
#                 [197, 185, 172],   # 1 = Unpaved Route
#                 [77, 69, 59],      # 2 = Paved Road
#                 [127, 122, 112],   # 3 = Tree Trunk
#                 [102, 127, 87],    # 4 = Tree Foliage
#                 [145, 145, 145],   # 7 = Rocks
#                 [158, 217, 92],    # 8 = Large Shrubs
#                 [200, 217, 92],    # 5 = Low Vegetation
#                 [216, 157, 92],    # 10 = Wire Fence
#                 [0, 0, 255],       # 9 = Sky
#                 [255, 0, 0],       # 11 = Person
#                 [53, 133, 193],    # 12 = Vehicle
#                 [0, 255, 0],       # 6 = Building
#                 [77, 69, 59],      # 2 = Paved Road
#                 [255, 0, 255],     # 13 = Misc
#                 [0, 185, 191],     # 14 = Water
#                 [65, 0, 74]   ,    # 15 = Animal
#                 [255, 255, 255],   # 16 =
#                 [255, 255, 255],   # 17 = Ignore
#      ]

####   Semantic_Extended     ####

PALETTE = [ [255, 0, 255],   # 0 = Clutter
                [0, 0, 0],   # 1 = Shadow
                [77, 69, 59],      # 2 = PavedRoad
                [197, 185, 172],   # 3 = UnpavedRoad
                [0, 128, 128],    # 4 = Pavement
                [0, 255, 0],   # 5 = Buildings
                [53, 133, 193],    # 6 = Cars
                [200, 217, 92],    # 7 = LowVegetation
                [102, 127, 87],    # 8 = HighVegetation
                [145, 145, 145],       # 9 = Rocks
                [148, 128, 106],       # 10 = Soil
                [0, 185, 191]    # 11 = Water
     ]


#####   Morphology    #####

# PALETTE = [[0, 0, 0],  # 0 = Unclassified
#                  [0, 64, 64],  # 2 = Clutter
#                  [20, 20, 20],  # 3 = Shadow
#                  [40, 40, 40],  # 5 = BrickWall
#                  [255, 255, 255],  # 10 = House
#                  [0, 128, 128],  # 15 = Car
#                  [0, 255, 255],  # 18 = Pavement
#                  [255, 0, 0],  # 20 = PavedRoad
#                  [128, 0, 255],  # 30 = DirtRoad
#                  [128, 0, 128],  # 35 = DirtRoadB
#                  [0, 128, 255],  # 40 = Water
#                  [0, 128, 0],  # 50 = Trees
#                  [0, 255, 0],  # 60 = LowVegetation
#                  [128, 128, 128],  # 70 = Rocks
#                  [128, 64, 0],  # 75 = RockySoil
#                  [128, 128, 0],  # 80 = DarkSoil
#                  [255, 128, 0],  # 85 = LightSoil
#                  [255, 255, 0],  # 90 = AgricalturalSoil
#                  [240, 222, 179],  # 92 = MaralRockyTerrain
#                  [243, 222, 179],  # 94 = MaralBadlands
#                  [244, 222, 179],  # 96 = MaralBoulder
#                  [238, 222, 179],  # 98 = MaralStoneyTerrain
#                  [242, 222, 179],  # 100 = MaralHardRockLineament
#                  [245, 222, 179],  # 102 = MaralBeddedRock
#                  [239, 222, 179],  # 104 = MaralSmoothRockSlopes
#                  [241, 222, 179],  # 106 = MaralRockDipSlope
#                  [246, 222, 179],  # 107 = MaralTerrace
#                  [237, 222, 179],  # 252 = MaralUndefined
#                  [243, 203, 173],  # 162 = LimestoneRockyTerrain
#                  [240, 203, 173],  # 164 = LimestoneBadlands
#                  [239, 203, 173],  # 166 = LimestoneBoulder
#                  [245, 203, 173],  # 168 = LimestoneStoneyTerrain
#                  [241, 203, 173],  # 170 = LimestoneHardRockLineament
#                  [238, 203, 173],  # 172 = LimestoneBeddedRock
#                  [244, 203, 173],  # 174 = LimestoneSmoothRockSlopes
#                  [242, 203, 173],  # 176 = LimestoneRockDipSlope
#                  [247, 203, 173],  # 177 = LimestoneTerrace
#                  [246, 203, 173],  # 242 = LimeStoneUndefined
#                  [144, 125, 107],  # 178 = DolomiteRockyTerrain
#                  [141, 125, 107],  # 180 = DolomiteBadlands
#                  [140, 125, 107],  # 182 = DolomiteBoulder
#                  [146, 125, 107],  # 184 = DolomiteStoneyTerrain
#                  [142, 125, 107],  # 186 = DolomiteHardRockLineament
#                  [139, 125, 107],  # 188 = DolomiteBeddedRock
#                  [145, 125, 107],  # 190 = DolomiteSmoothRockSlopes
#                  [143, 125, 107],  # 192 = DolomiteRockDipSlope
#                  [148, 125, 107],  # 193 = DolomiteTerrace
#                  [147, 125, 107],  # 244 = DolomiteUndefined
#                  [200, 197, 191],  # 194 = NariRockyTerrain
#                  [203, 197, 191],  # 196 = NariBadlands
#                  [204, 197, 191],  # 198 = NariBoulder
#                  [198, 197, 191],  # 200 = NariStoneyTerrain
#                  [202, 197, 191],  # 202 = NariHardRockLineament
#                  [205, 197, 191],  # 204 = NariBeddedRock
#                  [199, 197, 191],  # 206 = NariSmoothRockSlopes
#                  [201, 197, 191],  # 208 = NariRockDipSlope
#                  [206, 197, 191],  # 209 = NariTerrace
#                  [197, 197, 191],  # 246 = NariUndefined
#                  [99, 38, 18],  # 210 = BasaltRockyTerrain
#                  [96, 38, 18],  # 212 = BasaltBadlands
#                  [95, 38, 18],  # 214 = BasaltBoulder
#                  [101, 38, 18],  # 216 = BasaltStoneyTerrain
#                  [97, 38, 18],  # 218 = BasaltHardRockLineament
#                  [94, 38, 18],  # 220 = BasaltBeddedRock
#                  [100, 38, 18],  # 222 = BasaltSmoothRockSlopes
#                  [98, 38, 18],  # 224 = BasaltRockDipSlope
#                  [103, 38, 18],  # 225 = BasaltTerrace
#                  [102, 38, 18],  # 248 = BasaltUndefined
#                  [217, 184, 135],  # 226 = ChalkRockyTerrain
#                  [220, 184, 135],  # 228 = ChalkBadlands
#                  [221, 184, 135],  # 230 = ChalkBoulder
#                  [215, 184, 135],  # 232 = ChalkStoneyTerrain
#                  [219, 184, 135],  # 234 = ChalkHardRockLineament
#                  [222, 184, 135],  # 236 = ChalkBeddedRock
#                  [216, 184, 135],  # 238 = ChalkSmoothRockSlopes
#                  [218, 184, 135],  # 240 = ChalkRockDipSlope
#                  [223, 184, 135],  # 241 = ChalkTerrace
#                  [214, 184, 135],  # 250 = ChalkUndefined
#                  [218, 165, 32],  # 108 = TerraRosa
#                  [255, 193, 37],  # 110 = ClayeySoil
#                  [184, 115, 40],  # 111 = Colluvium
#                  [184, 134, 11],  # 112 = Rendzina
#                  [139, 105, 20],  # 114 = HydromorpicSoil
#                  [255, 165, 0],  # 116 = ClayeyDeepSoil
#                  [143, 188, 143],  # 118 = UnirrigatedOrchard
#                  [180, 238, 100],  # 120 = UnirrigatedField
#                  [105, 139, 105],  # 122 = IrrigatedOrchard
#                  [50, 205, 50],  # 124 = IrrigatedField
#                  [189, 252, 201],  # 126 = Batha
#                  [85, 107, 47],  # 128 = Garigue
#                  [34, 139, 34],  # 130 = Maquis
#                  [34, 100, 0],  # 131 = MaquisDense
#                  [125, 160, 35],  # 132 = DryGrassland
#                  [107, 142, 35],  # 12 = GreenGrassland
#                  ]
#
#
# ####   Convertion    ####
#
# convertion_palette={
# 0:0,        #Unclassified:               id=0:
# 1:2,      #Clutter:                    id=2:
# 2:3,     #Shadow:                     id=3:
# 3:10,  #House:                      id=10:
# 4:12,   #GreenGrassland:             id=12:
# 5:15,    #Car:                        id=15:
# 6:18,    # pavement                   id=18;
# 7:20,      #PavedRoad:                  id=20:
# 8:30,    #DirtRoad:                   id=30:
# 9:35,    #DirtRoadB:                  id=35:
# 10:40,   #Water:                      id=40:
# 11:108,  #TerraRosa:                  id=108:
# 12:110,  #ClayeySoil:                 id=110:
# 13:112,  #Rendzina:                   id=112:
# 14:116,   #ClayeyDeepSoil:             id=116:
# 15:118, #UnirrigatedOrchard:         id=118:
# 16:122, #IrrigatedOrchard:           id=122:
# 17:126, #Batha:                      id=126:
# 18:128,   #Garigue:                    id=128:
# 19:130,   #Maquis:                     id=130:
# 20:131,    #MaquisDense:                id=131:
# 21:132,  #DryGrassland:               id=132:
# 22:162, #LimestoneRockyTerrain:      id=162:
# 23:166, #LimestoneBoulder:           id=166:
# 24:168, #LimestoneStoneyTerrain:     id=168:
# 25:172, #LimestoneBeddedRock:        id=172:
# 26:174, #LimestoneSmoothRockSlopes:  id=174:
# 27:176, #LimestoneRockDipSlope:      id=176:
# 28:177, #LimestoneTerrace:           id=177:
# 29:178, #DolomiteRockyTerrain:       id=178:
# 30:182, #DolomiteBoulder:            id=182:
# 31:184, #DolomiteStoneyTerrain:      id=184:
# 32:194, #NariRockyTerrain:           id=194:
# 33:200, #NariStoneyTerrain:          id=200:
# 34:206, #NariSmoothRockSlopes:       id=206:
# 35:208, #NariRockDipSlope:           id=208:
# 36:210,    #BasaltRockyTerrain:         id=210:
# 37:214,    #BasaltBoulder:              id=214:
# 38:216,   #BasaltStoneyTerrain:        id=216:
# 39:238, #ChalkSmoothRockSlopes:      id=238:
# }
####   Cityscapes     ####

# PALETTE = [[128, 64, 128],
#            [244, 35, 232],
#            [70, 70, 70],
#            [102, 102, 156],
#            [190, 153, 153],
#            [153, 153, 153],
#            [250, 170, 30],
#            [220, 220, 0],
#            [107, 142, 35],
#            [152, 251, 152],
#            [70, 130, 180],
#            [220, 20, 60],
#            [255, 0, 0],
#            [0, 0, 142],
#            [0, 0, 70],
#            [0, 60, 100],
#            [0, 80, 100],
#            [0, 0, 230],
#            [119, 11, 32]]

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

    Seg_dir = path + '_Segmentation_slide'
    if not os.path.exists(Seg_dir):
        os.makedirs(Seg_dir)

    out_filename = os.path.join(Seg_dir , img_name)

    seg_logit = seg.seg_logits.data.cpu()
    seg_pred = seg.pred_sem_seg.data.cpu()#seg_logit.argmax(dim=0)
    ids = np.unique(seg_pred)[::-1]
    legal_indices = ids < num_classes
    ids = ids[legal_indices]
    labels = np.array(ids, dtype=np.int64)

    # colors = [palette[label] for label in labels]

    # seg_pred[seg_pred > 15] = 14
    seg_pred = np.squeeze(seg_pred)
    seg_pred = np.asarray(seg_pred,dtype=np.uint8)

    color_seg = np.zeros((seg_pred.shape[0], seg_pred.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[seg_pred == label, :] = color

    # # save the results
    im = Image.fromarray(color_seg)
    # converted_seg = np.zeros((seg_pred.shape[0], seg_pred.shape[1]), dtype=np.uint8)
    # for old_id in convertion_palette:
    #     converted_seg[seg_pred == old_id] = convertion_palette[old_id]

    # im = Image.fromarray(converted_seg)
    # im = im.resize((seg.ori_shape[1],seg.ori_shape[0]))
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
    images = list_images(args.path,ext='tif')

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
        save_segmentation(result, args.path, image_name, palette)
        # save_depth(result.pred_depth.data, image_name, palette)


if __name__ == '__main__':
    main()
