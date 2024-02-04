from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset
import os.path as osp
import mmengine
import mmengine.fileio as fileio

from clearml import Task


def nfs_to_clearml(nfs_path,nfs_prefix,clearml_prefix):
    return nfs_path.replace(nfs_prefix, clearml_prefix)

@DATASETS.register_module()
class MaterialsDataset(BaseSegDataset):
    #Our custom dataset.

    METAINFO = dict(
        classes=('Unclassified', 'Clutter', 'Shadow', 'BrickWall', 'House', 'Car',
                 'Pavement', 'PavedRoad', 'DirtRoad', 'DirtRoadB', 'Water', 'Trees', 'LowVegetation', 'Rocks',
                 'RockySoil', 'DarkSoil', 'LightSoil', 'AgricalturalSoil', 'Maral', 'Limestone', 'Dolomite', 'Nari', 'Basalt',
                 'Chalk', 'TerraRosa', 'ClayeySoil', 'Colluvium', 'Rendzina', 'HydromorpicSoil', 'ClayeyDeepSoil',
                 'StoneySoil', 'UnirrigatedOrchard', 'UnirrigatedField', 'IrrigatedOrchard', 'IrrigatedField',
                 'Batha', 'Garigue', 'Maquis', 'MaquisDense', 'DryGrassland', 'GreenGrassland'),
        palette=[[0, 0, 0],  # 0 = Unclassified
                 [0, 64, 64],  # 2 = Clutter
                 [20, 20, 20],  # 3 = Shadow
                 [40, 40, 40],  # 5 = BrickWall
                 [255, 255, 255],  # 10 = House
                 [0, 128, 128],  # 15 = Car
                 [0, 255, 255],  # 18 = Pavement
                 [255, 0, 0],  # 20 = PavedRoad
                 [128, 0, 255],  # 30 = DirtRoad
                 [128, 0, 128],  # 35 = DirtRoadB
                 [0, 128, 255],  # 40 = Water
                 [0, 128, 0],  # 50 = Trees
                 [0, 255, 0],  # 60 = LowVegetation
                 [128, 128, 128],  # 70 = Rocks
                 [128, 64, 0],  # 75 = RockySoil
                 [128, 128, 0],  # 80 = DarkSoil
                 [255, 128, 0],  # 85 = LightSoil
                 [255, 255, 0],  # 90 = AgricalturalSoil
                 [240, 222, 179],  # 92 = Maral
                 [243, 203, 173],  # 162 = Limestone
                 [144, 125, 107],  # 178 = Dolomite
                 [200, 197, 191],  # 194 = Nari
                 [99, 38, 18],  # 210 = Basalt
                 [217, 184, 135],  # 226 = Chalk
                 [218, 165, 32],  # 108 = TerraRosa
                 [255, 193, 37],  # 110 = ClayeySoil
                 [184, 115, 40],  # 111 = Colluvium
                 [184, 134, 11],  # 112 = Rendzina
                 [139, 105, 20],  # 114 = HydromorpicSoil
                 [255, 165, 0],  # 116 = ClayeyDeepSoil
                 [184, 100, 0],  # 117 = StoneySoil
                 [143, 188, 143],  # 118 = UnirrigatedOrchard
                 [180, 238, 100],  # 120 = UnirrigatedField
                 [105, 139, 105],  # 122 = IrrigatedOrchard
                 [50, 205, 50],  # 124 = IrrigatedField
                 [189, 252, 201],  # 126 = Batha
                 [85, 107, 47],  # 128 = Garigue
                 [34, 139, 34],  # 130 = Maquis
                 [34, 100, 0],  # 131 = MaquisDense
                 [125, 160, 35],  # 132 = DryGrassland
                 [107, 142, 35],  # 12 = GreenGrassland
                 ])

    CLASSES = ('Unclassified', 'Clutter', 'Shadow', 'BrickWall', 'House', 'Car',
               'Pavement', 'PavedRoad', 'DirtRoad', 'DirtRoadB', 'Water', 'Trees', 'LowVegetation', 'Rocks',
               'RockySoil', 'DarkSoil', 'LightSoil', 'AgricalturalSoil', 'Maral', 'Limestone', 'Dolomite', 'Nari', 'Basalt',
               'Chalk', 'TerraRosa', 'ClayeySoil', 'Colluvium', 'Rendzina', 'HydromorpicSoil', 'ClayeyDeepSoil',
               'StoneySoil', 'UnirrigatedOrchard', 'UnirrigatedField', 'IrrigatedOrchard', 'IrrigatedField',
               'Batha', 'Garigue', 'Maquis', 'MaquisDense', 'DryGrassland', 'GreenGrassland')

    PALETTE = [[0, 0, 0],  # 0 = Unclassified
               [0, 64, 64],  # 2 = Clutter
               [20, 20, 20],  # 3 = Shadow
               [40, 40, 40],  # 5 = BrickWall
               [255, 255, 255],  # 10 = House
               [0, 128, 128],  # 15 = Car
               [0, 255, 255],  # 18 = Pavement
               [255, 0, 0],  # 20 = PavedRoad
               [128, 0, 255],  # 30 = DirtRoad
               [128, 0, 128],  # 35 = DirtRoadB
               [0, 128, 255],  # 40 = Water
               [0, 128, 0],  # 50 = Trees
               [0, 255, 0],  # 60 = LowVegetation
               [128, 128, 128],  # 70 = Rocks
               [128, 64, 0],  # 75 = RockySoil
               [128, 128, 0],  # 80 = DarkSoil
               [255, 128, 0],  # 85 = LightSoil
               [255, 255, 0],  # 90 = AgricalturalSoil
               [240, 222, 179],  # 92 = Maral
               [243, 203, 173],  # 162 = Limestone
               [144, 125, 107],  # 178 = Dolomite
               [200, 197, 191],  # 194 = Nari
               [99, 38, 18],  # 210 = Basalt
               [217, 184, 135],  # 226 = Chalk
               [218, 165, 32],  # 108 = TerraRosa
               [255, 193, 37],  # 110 = ClayeySoil
               [184, 115, 40],  # 111 = Colluvium
               [184, 134, 11],  # 112 = Rendzina
               [139, 105, 20],  # 114 = HydromorpicSoil
               [255, 165, 0],  # 116 = ClayeyDeepSoil
               [184, 100, 0],  # 117 = StoneySoil
               [143, 188, 143],  # 118 = UnirrigatedOrchard
               [180, 238, 100],  # 120 = UnirrigatedField
               [105, 139, 105],  # 122 = IrrigatedOrchard
               [50, 205, 50],  # 124 = IrrigatedField
               [189, 252, 201],  # 126 = Batha
               [85, 107, 47],  # 128 = Garigue
               [34, 139, 34],  # 130 = Maquis
               [34, 100, 0],  # 131 = MaquisDense
               [125, 160, 35],  # 132 = DryGrassland
               [107, 142, 35],  # 12 = GreenGrassland
   ]

    def __init__(self, **kwargs):
        super(MaterialsDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            **kwargs)

        self.nfs_prefix = '/hdd_data/asaf/ido_test/AllData_split'  # à prefix to my local path
        self.clearml_prefix = '/data'
        self.execute_remotly = False
        self.prefetch = False
        # all_dvs = Task.current_task().get_dataviews()
        # dv = all_dvs.get('Materials')
        #
        # if self.prefetch:
        #     print("Prefetching Data")
        #     dv.prefetch_files(num_workers=10, wait=True)

    def load_data_list(self):
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []

        if osp.isfile(self.ann_file):
            lines = mmengine.list_from_file(
                self.ann_file, backend_args=self.backend_args)
            for line in lines:
                # if self.execute_remotly:
                # line = nfs_to_clearml(line,'/hdd_data/asaf/ido_test/AllData_split','/data')

                # IMAGE
                img_name = line.strip().replace('Labels_mask_Material', 'Ortho_RGB')
                data_info = dict(
                    img_path=img_name)
                # LABEL
                seg_map = line.strip()
                data_info['seg_map_path'] = seg_map
                data_info['label_map'] = self.label_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []
                data_list.append(data_info)
        else:
            print('Expected a split file.')
        return data_list

@DATASETS.register_module()
class SemanticDataset(BaseSegDataset):
    #Our custom dataset.

    METAINFO = dict(
        classes=('ImperviousSurfaces', 'Buildings', 'LowVegetation', 'Trees', 'Cars', 'Clutter',
               'Rocks', 'Soil', 'Shadow','Water'),
        palette=[ [255, 255, 255],   # 0 = ImperviousSurfaces
                [0, 0, 255],   # 1 = Buildings
                [0, 255, 255],      # 2 = LowVegetation
                [0, 255, 0],   # 3 = Trees
                [255, 255, 0],    # 4 = Cars
                [255, 0, 0],   # 5 = Clutter
                [0, 0, 0],    # 6 = Rocks
                [222, 184, 135],    # 7 = Soil
                [125, 125, 125],    # 8 = Shadow
                [0, 128, 128],       # 9 = Water
    ])

    CLASSES = ('ImperviousSurfaces', 'Buildings', 'LowVegetation', 'Trees', 'Cars', 'Clutter',
               'Rocks', 'Soil', 'Shadow','Water')

    PALETTE = [ [255, 255, 255],   # 0 = ImperviousSurfaces
                [0, 0, 255],   # 1 = Buildings
                [0, 255, 255],      # 2 = LowVegetation
                [0, 255, 0],   # 3 = Trees
                [255, 255, 0],    # 4 = Cars
                [255, 0, 0],   # 5 = Clutter
                [0, 0, 0],    # 6 = Rocks
                [222, 184, 135],    # 7 = Soil
                [125, 125, 125],    # 8 = Shadow
                [0, 128, 128]  # 9 = Water
    ]

    def __init__(self, **kwargs):
        super(SemanticDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            **kwargs)

        self.nfs_prefix = '/hdd_data/asaf/ido_test/AllData_split'  # à prefix to my local path
        self.clearml_prefix = '/data'
        self.execute_remotly = False
        self.prefetch = False
        # all_dvs = Task.current_task().get_dataviews()
        # dv = all_dvs.get('Semantic')
        #
        # if self.prefetch:
        #     print("Prefetching Data")
        #     dv.prefetch_files(num_workers=10, wait=True)
    def load_data_list(self):
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []
        if osp.isfile(self.ann_file):
            lines = mmengine.list_from_file(
                self.ann_file, backend_args=self.backend_args)
            for line in lines:
                # if self.execute_remotly:
                # line = nfs_to_clearml(line,'/hdd_data/asaf/ido_test/AllData_split','/data')
                # IMAGE
                img_name = line.strip().replace('Labels_mask_Semantic', 'Ortho_RGB')
                data_info = dict(
                    img_path=img_name)
                # LABEL
                seg_map = line.strip()
                data_info['seg_map_path'] = seg_map
                data_info['label_map'] = self.label_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []
                data_list.append(data_info)
        else:
            print('Expected a split file.')
        return data_list

@DATASETS.register_module()
class SemanticExtendedDataset(BaseSegDataset):
    #Our custom dataset.

    METAINFO = dict(
        classes=('Clutter', 'Shadow', 'PavedRoad', 'UnpavedRoad', 'Pavement', 'Buildings',
               'Cars', 'LowVegetation', 'HighVegetation', 'Rocks', 'Soil',
               'Water'),
        palette= [ [255, 0, 255],   # 0 = Clutter
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
     ])

    CLASSES = ('Clutter', 'Shadow', 'PavedRoad', 'UnpavedRoad', 'Pavement', 'Buildings',
               'Cars', 'LowVegetation', 'HighVegetation', 'Rocks', 'Soil',
               'Water')

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

    def __init__(self, **kwargs):
        super(SemanticExtendedDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            **kwargs)

        self.nfs_prefix = '/hdd_data/asaf/ido_test/AllData_split'  # à prefix to my local path
        self.clearml_prefix = '/data'
        self.execute_remotly = False
        self.prefetch = False
        # all_dvs = Task.current_task().get_dataviews()
        # dv = all_dvs.get('SemanticExtended')
        #
        # if self.prefetch:
        #     print("Prefetching Data")
        #     dv.prefetch_files(num_workers=10, wait=True)

    def load_data_list(self):
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []
        if osp.isfile(self.ann_file):
            lines = mmengine.list_from_file(
                self.ann_file, backend_args=self.backend_args)
            for line in lines:
                # if self.execute_remotly:
                # line = nfs_to_clearml(line,'/hdd_data/asaf/ido_test/AllData_split','/data')
                # IMAGE
                img_name = line.strip().replace('Labels_mask_SemanticExtended', 'Ortho_RGB')
                data_info = dict(
                    img_path=img_name)
                # LABEL
                seg_map = line.strip()
                data_info['seg_map_path'] = seg_map

                data_info['label_map'] = self.label_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []
                data_list.append(data_info)
        else:
            print('Expected a split file.')
        return data_list


@DATASETS.register_module()
class RoadsDataset(BaseSegDataset):
    #Our custom dataset.

    METAINFO = dict(
        classes=('Background', 'Road'),
        palette= [ [0, 0, 0],   # 0 = Background
                [255, 0, 0] # 1 = Road
     ])

    CLASSES = ('Background', 'Road')

    PALETTE = [ [0, 0, 0],   # 0 = Background
                [255, 0, 0] # 1 = Road
     ]

    def __init__(self, **kwargs):
        super(RoadsDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            **kwargs)

    def load_data_list(self):
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []
        if osp.isfile(self.ann_file):
            lines = mmengine.list_from_file(
                self.ann_file, backend_args=self.backend_args)
            for line in lines:
                # if self.execute_remotly:
                # line = nfs_to_clearml(line,'/hdd_data/asaf/ido_test/AllData_split','/data')
                # IMAGE
                img_name = line.strip().replace('Labels_mask_Roads', 'Ortho_RGB')
                data_info = dict(
                    img_path=img_name)
                # LABEL
                seg_map = line.strip()
                data_info['seg_map_path'] = seg_map

                data_info['label_map'] = self.label_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []
                data_list.append(data_info)
        else:
            print('Expected a split file.')
        return data_list

@DATASETS.register_module()
class MorphologyDataset(BaseSegDataset):
    #Our custom dataset.

    METAINFO = dict(
        classes=('Unclassified', 'Clutter', 'Shadow', 'BrickWall', 'House', 'Car',
                 'Pavement', 'PavedRoad', 'DirtRoad', 'DirtRoadB', 'Water', 'Trees', 'LowVegetation', 'Rocks',
                 'RockySoil', 'DarkSoil', 'LightSoil', 'AgricalturalSoil', 'MaralRockyTerrain', 'MaralBadlands',
                 'MaralBoulder', 'MaralStoneyTerrain', 'MaralHardRockLineament', 'MaralBeddedRock', 'MaralSmoothRockSlopes',
                 'MaralRockDipSlope', 'MaralTerrace', 'MaralUndefined', 'LimestoneRockyTerrain', 'LimestoneBadlands',
                 'LimestoneBoulder', 'LimestoneStoneyTerrain', 'LimestoneHardRockLineament', 'LimestoneBeddedRock',
                 'LimestoneSmoothRockSlopes', 'LimestoneRockDipSlope', 'LimestoneTerrace', 'LimeStoneUndefined',
                 'DolomiteRockyTerrain', 'DolomiteBadlands', 'DolomiteBoulder', 'DolomiteStoneyTerrain', 'DolomiteHardRockLineament',
                 'DolomiteBeddedRock', 'DolomiteSmoothRockSlopes', 'DolomiteRockDipSlope', 'DolomiteTerrace',
                 'DolomiteUndefined', 'NariRockyTerrain', 'NariBadlands', 'NariBoulder', 'NariStoneyTerrain', 'NariHardRockLineament',
                 'NariBeddedRock', 'NariSmoothRockSlopes', 'NariRockDipSlope', 'NariTerrace', 'NariUndefined',
                 'BasaltRockyTerrain', 'BasaltBadlands', 'BasaltBoulder', 'BasaltStoneyTerrain', 'BasaltHardRockLineament',
                 'BasaltBeddedRock', 'BasaltSmoothRockSlopes', 'BasaltRockDipSlope', 'BasaltTerrace', 'BasaltUndefined',
                 'ChalkRockyTerrain', 'ChalkBadlands', 'ChalkBoulder', 'ChalkStoneyTerrain', 'ChalkHardRockLineament',
                 'ChalkBeddedRock', 'ChalkSmoothRockSlopes', 'ChalkRockDipSlope', 'ChalkTerrace', 'ChalkUndefined',
                 'TerraRosa', 'ClayeySoil', 'Colluvium', 'Rendzina', 'HydromorpicSoil', 'ClayeyDeepSoil', 'UnirrigatedOrchard',
                 'UnirrigatedField', 'IrrigatedOrchard', 'IrrigatedField', 'Batha', 'Garigue', 'Maquis',
                 'MaquisDense', 'DryGrassland', 'GreenGrassland'),
        palette=[[0, 0, 0],  # 0 = Unclassified
                 [0, 64, 64],  # 2 = Clutter
                 [20, 20, 20],  # 3 = Shadow
                 [40, 40, 40],  # 5 = BrickWall
                 [255, 255, 255],  # 10 = House
                 [0, 128, 128],  # 15 = Car
                 [0, 255, 255],  # 18 = Pavement
                 [255, 0, 0],  # 20 = PavedRoad
                 [128, 0, 255],  # 30 = DirtRoad
                 [128, 0, 128],  # 35 = DirtRoadB
                 [0, 128, 255],  # 40 = Water
                 [0, 128, 0],  # 50 = Trees
                 [0, 255, 0],  # 60 = LowVegetation
                 [128, 128, 128],  # 70 = Rocks
                 [128, 64, 0],  # 75 = RockySoil
                 [128, 128, 0],  # 80 = DarkSoil
                 [255, 128, 0],  # 85 = LightSoil
                 [255, 255, 0],  # 90 = AgricalturalSoil
                 [240, 222, 179],  # 92 = MaralRockyTerrain
                 [243, 222, 179],  # 94 = MaralBadlands
                 [244, 222, 179],  # 96 = MaralBoulder
                 [238, 222, 179],  # 98 = MaralStoneyTerrain
                 [242, 222, 179],  # 100 = MaralHardRockLineament
                 [245, 222, 179],  # 102 = MaralBeddedRock
                 [239, 222, 179],  # 104 = MaralSmoothRockSlopes
                 [241, 222, 179],  # 106 = MaralRockDipSlope
                 [246, 222, 179],  # 107 = MaralTerrace
                 [237, 222, 179],  # 252 = MaralUndefined
                 [243, 203, 173],  # 162 = LimestoneRockyTerrain
                 [240, 203, 173],  # 164 = LimestoneBadlands
                 [239, 203, 173],  # 166 = LimestoneBoulder
                 [245, 203, 173],  # 168 = LimestoneStoneyTerrain
                 [241, 203, 173],  # 170 = LimestoneHardRockLineament
                 [238, 203, 173],  # 172 = LimestoneBeddedRock
                 [244, 203, 173],  # 174 = LimestoneSmoothRockSlopes
                 [242, 203, 173],  # 176 = LimestoneRockDipSlope
                 [247, 203, 173],  # 177 = LimestoneTerrace
                 [246, 203, 173],  # 242 = LimeStoneUndefined
                 [144, 125, 107],  # 178 = DolomiteRockyTerrain
                 [141, 125, 107],  # 180 = DolomiteBadlands
                 [140, 125, 107],  # 182 = DolomiteBoulder
                 [146, 125, 107],  # 184 = DolomiteStoneyTerrain
                 [142, 125, 107],  # 186 = DolomiteHardRockLineament
                 [139, 125, 107],  # 188 = DolomiteBeddedRock
                 [145, 125, 107],  # 190 = DolomiteSmoothRockSlopes
                 [143, 125, 107],  # 192 = DolomiteRockDipSlope
                 [148, 125, 107],  # 193 = DolomiteTerrace
                 [147, 125, 107],  # 244 = DolomiteUndefined
                 [200, 197, 191],  # 194 = NariRockyTerrain
                 [203, 197, 191],  # 196 = NariBadlands
                 [204, 197, 191],  # 198 = NariBoulder
                 [198, 197, 191],  # 200 = NariStoneyTerrain
                 [202, 197, 191],  # 202 = NariHardRockLineament
                 [205, 197, 191],  # 204 = NariBeddedRock
                 [199, 197, 191],  # 206 = NariSmoothRockSlopes
                 [201, 197, 191],  # 208 = NariRockDipSlope
                 [206, 197, 191],  # 209 = NariTerrace
                 [197, 197, 191],  # 246 = NariUndefined
                 [99, 38, 18],  # 210 = BasaltRockyTerrain
                 [96, 38, 18],  # 212 = BasaltBadlands
                 [95, 38, 18],  # 214 = BasaltBoulder
                 [101, 38, 18],  # 216 = BasaltStoneyTerrain
                 [97, 38, 18],  # 218 = BasaltHardRockLineament
                 [94, 38, 18],  # 220 = BasaltBeddedRock
                 [100, 38, 18],  # 222 = BasaltSmoothRockSlopes
                 [98, 38, 18],  # 224 = BasaltRockDipSlope
                 [103, 38, 18],  # 225 = BasaltTerrace
                 [102, 38, 18],  # 248 = BasaltUndefined
                 [217, 184, 135],  # 226 = ChalkRockyTerrain
                 [220, 184, 135],  # 228 = ChalkBadlands
                 [221, 184, 135],  # 230 = ChalkBoulder
                 [215, 184, 135],  # 232 = ChalkStoneyTerrain
                 [219, 184, 135],  # 234 = ChalkHardRockLineament
                 [222, 184, 135],  # 236 = ChalkBeddedRock
                 [216, 184, 135],  # 238 = ChalkSmoothRockSlopes
                 [218, 184, 135],  # 240 = ChalkRockDipSlope
                 [223, 184, 135],  # 241 = ChalkTerrace
                 [214, 184, 135],  # 250 = ChalkUndefined
                 [218, 165, 32],  # 108 = TerraRosa
                 [255, 193, 37],  # 110 = ClayeySoil
                 [184, 115, 40],  # 111 = Colluvium
                 [184, 134, 11],  # 112 = Rendzina
                 [139, 105, 20],  # 114 = HydromorpicSoil
                 [255, 165, 0],  # 116 = ClayeyDeepSoil
                 [143, 188, 143],  # 118 = UnirrigatedOrchard
                 [180, 238, 100],  # 120 = UnirrigatedField
                 [105, 139, 105],  # 122 = IrrigatedOrchard
                 [50, 205, 50],  # 124 = IrrigatedField
                 [189, 252, 201],  # 126 = Batha
                 [85, 107, 47],  # 128 = Garigue
                 [34, 139, 34],  # 130 = Maquis
                 [34, 100, 0],  # 131 = MaquisDense
                 [125, 160, 35],  # 132 = DryGrassland
                 [107, 142, 35],  # 12 = GreenGrassland
                 ])

    CLASSES = ('Unclassified', 'Clutter', 'Shadow', 'BrickWall', 'House', 'Car',
                 'Pavement', 'PavedRoad', 'DirtRoad', 'DirtRoadB', 'Water', 'Trees', 'LowVegetation', 'Rocks',
                 'RockySoil', 'DarkSoil', 'LightSoil', 'AgricalturalSoil', 'MaralRockyTerrain', 'MaralBadlands',
                 'MaralBoulder', 'MaralStoneyTerrain', 'MaralHardRockLineament', 'MaralBeddedRock', 'MaralSmoothRockSlopes',
                 'MaralRockDipSlope', 'MaralTerrace', 'MaralUndefined', 'LimestoneRockyTerrain', 'LimestoneBadlands',
                 'LimestoneBoulder', 'LimestoneStoneyTerrain', 'LimestoneHardRockLineament', 'LimestoneBeddedRock',
                 'LimestoneSmoothRockSlopes', 'LimestoneRockDipSlope', 'LimestoneTerrace', 'LimeStoneUndefined',
                 'DolomiteRockyTerrain', 'DolomiteBadlands', 'DolomiteBoulder', 'DolomiteStoneyTerrain', 'DolomiteHardRockLineament',
                 'DolomiteBeddedRock', 'DolomiteSmoothRockSlopes', 'DolomiteRockDipSlope', 'DolomiteTerrace',
                 'DolomiteUndefined', 'NariRockyTerrain', 'NariBadlands', 'NariBoulder', 'NariStoneyTerrain', 'NariHardRockLineament',
                 'NariBeddedRock', 'NariSmoothRockSlopes', 'NariRockDipSlope', 'NariTerrace', 'NariUndefined',
                 'BasaltRockyTerrain', 'BasaltBadlands', 'BasaltBoulder', 'BasaltStoneyTerrain', 'BasaltHardRockLineament',
                 'BasaltBeddedRock', 'BasaltSmoothRockSlopes', 'BasaltRockDipSlope', 'BasaltTerrace', 'BasaltUndefined',
                 'ChalkRockyTerrain', 'ChalkBadlands', 'ChalkBoulder', 'ChalkStoneyTerrain', 'ChalkHardRockLineament',
                 'ChalkBeddedRock', 'ChalkSmoothRockSlopes', 'ChalkRockDipSlope', 'ChalkTerrace', 'ChalkUndefined',
                 'TerraRosa', 'ClayeySoil', 'Colluvium', 'Rendzina', 'HydromorpicSoil', 'ClayeyDeepSoil', 'UnirrigatedOrchard',
                 'UnirrigatedField', 'IrrigatedOrchard', 'IrrigatedField', 'Batha', 'Garigue', 'Maquis',
                 'MaquisDense', 'DryGrassland', 'GreenGrassland')

    PALETTE = [[0, 0, 0],  # 0 = Unclassified
                 [0, 64, 64],  # 2 = Clutter
                 [20, 20, 20],  # 3 = Shadow
                 [40, 40, 40],  # 5 = BrickWall
                 [255, 255, 255],  # 10 = House
                 [0, 128, 128],  # 15 = Car
                 [0, 255, 255],  # 18 = Pavement
                 [255, 0, 0],  # 20 = PavedRoad
                 [128, 0, 255],  # 30 = DirtRoad
                 [128, 0, 128],  # 35 = DirtRoadB
                 [0, 128, 255],  # 40 = Water
                 [0, 128, 0],  # 50 = Trees
                 [0, 255, 0],  # 60 = LowVegetation
                 [128, 128, 128],  # 70 = Rocks
                 [128, 64, 0],  # 75 = RockySoil
                 [128, 128, 0],  # 80 = DarkSoil
                 [255, 128, 0],  # 85 = LightSoil
                 [255, 255, 0],  # 90 = AgricalturalSoil
                 [240, 222, 179],  # 92 = MaralRockyTerrain
                 [243, 222, 179],  # 94 = MaralBadlands
                 [244, 222, 179],  # 96 = MaralBoulder
                 [238, 222, 179],  # 98 = MaralStoneyTerrain
                 [242, 222, 179],  # 100 = MaralHardRockLineament
                 [245, 222, 179],  # 102 = MaralBeddedRock
                 [239, 222, 179],  # 104 = MaralSmoothRockSlopes
                 [241, 222, 179],  # 106 = MaralRockDipSlope
                 [246, 222, 179],  # 107 = MaralTerrace
                 [237, 222, 179],  # 252 = MaralUndefined
                 [243, 203, 173],  # 162 = LimestoneRockyTerrain
                 [240, 203, 173],  # 164 = LimestoneBadlands
                 [239, 203, 173],  # 166 = LimestoneBoulder
                 [245, 203, 173],  # 168 = LimestoneStoneyTerrain
                 [241, 203, 173],  # 170 = LimestoneHardRockLineament
                 [238, 203, 173],  # 172 = LimestoneBeddedRock
                 [244, 203, 173],  # 174 = LimestoneSmoothRockSlopes
                 [242, 203, 173],  # 176 = LimestoneRockDipSlope
                 [247, 203, 173],  # 177 = LimestoneTerrace
                 [246, 203, 173],  # 242 = LimeStoneUndefined
                 [144, 125, 107],  # 178 = DolomiteRockyTerrain
                 [141, 125, 107],  # 180 = DolomiteBadlands
                 [140, 125, 107],  # 182 = DolomiteBoulder
                 [146, 125, 107],  # 184 = DolomiteStoneyTerrain
                 [142, 125, 107],  # 186 = DolomiteHardRockLineament
                 [139, 125, 107],  # 188 = DolomiteBeddedRock
                 [145, 125, 107],  # 190 = DolomiteSmoothRockSlopes
                 [143, 125, 107],  # 192 = DolomiteRockDipSlope
                 [148, 125, 107],  # 193 = DolomiteTerrace
                 [147, 125, 107],  # 244 = DolomiteUndefined
                 [200, 197, 191],  # 194 = NariRockyTerrain
                 [203, 197, 191],  # 196 = NariBadlands
                 [204, 197, 191],  # 198 = NariBoulder
                 [198, 197, 191],  # 200 = NariStoneyTerrain
                 [202, 197, 191],  # 202 = NariHardRockLineament
                 [205, 197, 191],  # 204 = NariBeddedRock
                 [199, 197, 191],  # 206 = NariSmoothRockSlopes
                 [201, 197, 191],  # 208 = NariRockDipSlope
                 [206, 197, 191],  # 209 = NariTerrace
                 [197, 197, 191],  # 246 = NariUndefined
                 [99, 38, 18],  # 210 = BasaltRockyTerrain
                 [96, 38, 18],  # 212 = BasaltBadlands
                 [95, 38, 18],  # 214 = BasaltBoulder
                 [101, 38, 18],  # 216 = BasaltStoneyTerrain
                 [97, 38, 18],  # 218 = BasaltHardRockLineament
                 [94, 38, 18],  # 220 = BasaltBeddedRock
                 [100, 38, 18],  # 222 = BasaltSmoothRockSlopes
                 [98, 38, 18],  # 224 = BasaltRockDipSlope
                 [103, 38, 18],  # 225 = BasaltTerrace
                 [102, 38, 18],  # 248 = BasaltUndefined
                 [217, 184, 135],  # 226 = ChalkRockyTerrain
                 [220, 184, 135],  # 228 = ChalkBadlands
                 [221, 184, 135],  # 230 = ChalkBoulder
                 [215, 184, 135],  # 232 = ChalkStoneyTerrain
                 [219, 184, 135],  # 234 = ChalkHardRockLineament
                 [222, 184, 135],  # 236 = ChalkBeddedRock
                 [216, 184, 135],  # 238 = ChalkSmoothRockSlopes
                 [218, 184, 135],  # 240 = ChalkRockDipSlope
                 [223, 184, 135],  # 241 = ChalkTerrace
                 [214, 184, 135],  # 250 = ChalkUndefined
                 [218, 165, 32],  # 108 = TerraRosa
                 [255, 193, 37],  # 110 = ClayeySoil
                 [184, 115, 40],  # 111 = Colluvium
                 [184, 134, 11],  # 112 = Rendzina
                 [139, 105, 20],  # 114 = HydromorpicSoil
                 [255, 165, 0],  # 116 = ClayeyDeepSoil
                 [143, 188, 143],  # 118 = UnirrigatedOrchard
                 [180, 238, 100],  # 120 = UnirrigatedField
                 [105, 139, 105],  # 122 = IrrigatedOrchard
                 [50, 205, 50],  # 124 = IrrigatedField
                 [189, 252, 201],  # 126 = Batha
                 [85, 107, 47],  # 128 = Garigue
                 [34, 139, 34],  # 130 = Maquis
                 [34, 100, 0],  # 131 = MaquisDense
                 [125, 160, 35],  # 132 = DryGrassland
                 [107, 142, 35],  # 12 = GreenGrassland
                 ]

    def __init__(self, **kwargs):
        super(MorphologyDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            **kwargs)

        self.nfs_prefix = '/hdd_data/asaf/ido_test/AllData_split'  # à prefix to my local path
        self.clearml_prefix = '/data'
        self.execute_remotly = False
        self.prefetch = False
        # all_dvs = Task.current_task().get_dataviews()
        # dv = all_dvs.get('Morphology')
        #
        # if self.prefetch:
        #     print("Prefetching Data")
        #     dv.prefetch_files(num_workers=10, wait=True)

    def load_data_list(self):
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []
        if osp.isfile(self.ann_file):
            lines = mmengine.list_from_file(
                self.ann_file, backend_args=self.backend_args)
            for line in lines:
                # if self.execute_remotly:
                # line = nfs_to_clearml(line,'/hdd_data/asaf/ido_test/AllData_split','/data')
                # IMAGE
                img_name = line.strip().replace('Labels_mask_MaterialMorphology', 'Ortho_RGB')
                data_info = dict(
                    img_path=img_name)
                # LABEL
                seg_map = line.strip()
                data_info['seg_map_path'] = seg_map
                data_info['label_map'] = self.label_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []
                data_list.append(data_info)
        else:
            print('Expected a split file.')
        return data_list



