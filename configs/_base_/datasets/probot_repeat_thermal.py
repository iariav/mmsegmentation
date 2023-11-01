# dataset settings
dataset_type = 'ProbotDataset'
# dataset_type = 'ProbotDatasetClear'
data_root = '/ssd_data/NAS/thermal/'
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (504, 504)
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        color_type='unchanged',
        imdecode_backend='cv2'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(1280, 720),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        color_type='unchanged',
        imdecode_backend='cv2'),
    dict(type='Resize', scale=(1280, 720), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    dataset=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            data_prefix = dict(img_path='ThermalImages', seg_map_path='ThermalLabels'),
            ann_file = 'trainval.txt',
            pipeline=train_pipeline)),
    sampler=dict(type='DefaultSampler', shuffle=True)  # necessary
)

val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix = dict(img_path='ThermalImages', seg_map_path='ThermalLabels'),
        ann_file = 'val.txt',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False)  # necessary
)

test_dataloader = val_dataloader
