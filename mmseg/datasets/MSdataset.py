from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset
import os.path as osp
import mmengine
import mmengine.fileio as fileio

@DATASETS.register_module()
class MSDepthDataset(BaseSegDataset):
    #Our custom dataset.

    METAINFO = dict(
        classes=('Terrain', 'Unpaved Route', 'Paved Road', 'Tree Trunk', 'Tree Foliage', 'Rocks',
               'Large Shrubs', 'Low Vegetation', 'Wire Fence', 'Sky', 'Person',
               'Vehicle', 'Building', 'Paved Road', 'Misc', 'Water', 'Animal', 'Ignore',
               'Ignore'),
        palette=[ [148, 128, 106],   # 0 = Terrain
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
    ])

    CLASSES = ('Terrain', 'Unpaved Route', 'Paved Road', 'Tree Trunk', 'Tree Foliage', 'Rocks',
               'Large Shrubs', 'Low Vegetation', 'Wire Fence', 'Sky', 'Person',
               'Vehicle', 'Building', 'Paved Road', 'Misc', 'Water', 'Animal', 'Ignore',
               'Ignore')

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

    def __init__(self,
                 img_suffix='.png',
                 min_depth=1e-3,
                 max_depth=65.0,
                 depth_scale=1,
                 **kwargs):

        self.min_depth = min_depth
        self.max_depth = max_depth
        self.depth_scale = depth_scale

        super(MSDepthDataset, self).__init__(
            img_suffix='.png',
            **kwargs)

    def _get_image_filename(self, depth_file):
        """Get the corresponding depth file from an image file."""
        image_file = depth_file.replace('depth_filtered', 'img_left').replace('depth', 'img_left').replace('npz', 'png')

        return image_file
    def load_data_list(self):
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []
        # img_dir = self.data_prefix.get('img_path', None)
        # ann_dir = self.data_prefix.get('seg_map_path', None)
        if osp.isfile(self.ann_file):
            lines = mmengine.list_from_file(
                self.ann_file, backend_args=self.backend_args)
            for line in lines:
                # IMAGE
                img_name = self._get_image_filename(line.strip())
                data_info = dict(
                    img_path=img_name)
                # DEPTH
                depth_name = line.strip()
                if osp.exists(depth_name):
                    data_info['depth_map_path'] = depth_name
                    data_info['depth_fields'] = []
                    data_info['depth_scale'] = self.depth_scale
                    data_info['min_depth'] = self.min_depth
                    data_info['max_depth'] = self.max_depth


                # data_info['label_map'] = self.label_map
                # data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []
                data_list.append(data_info)
        else:
            print('Expected a split file.')
        return data_list