# dataset settings
dataset_type = 'ProbotDataset'
# dataset_type = 'ProbotDatasetClear'
data_root = '/ssd_data/SegmentationTrainingData/'
depth_data_root = '/ssd_data/DepthTrainingData'
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (504, 504)

#######       DATA PIPELINES       #######
train_seg_pipeline = [
    dict(type='LoadImageFromFile'),
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

train_depth_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='DepthLoadAnnotationsNpy'),
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
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=(1280, 720), keep_ratio=True),
    dict(type='PackSegInputs')
]

img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]

test_depth_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='DepthLoadAnnotationsNpy'),
    dict(type='Resize', scale=(1280, 720), keep_ratio=True),
    dict(type='PackSegInputs')
]

#######       DATASETS       #######

### DEPTH ###
dataset_probot_depth_train = dict(
    type='RepeatDataset',
    times=10,
    dataset=dict(
        type='ProbotDepthDataset',
        data_root=depth_data_root,
        ann_file='DepthProbot_train.txt',
        pipeline=train_depth_pipeline,
        depth_scale=1000,
        min_depth=1e-2,
        max_depth=65
    )
)

dataset_airsim_depth_train = dict(
    type='RepeatDataset',
    times=1,
    dataset=dict(
        type='ProbotDepthDataset',
        data_root=depth_data_root,
        ann_file='DepthProbot_Airsim_train.txt',
        pipeline=train_depth_pipeline,
        depth_scale=1,
        min_depth=1e-2,
        max_depth=65
    )
)

# dataset_ddad_depth_train = dict(
#     type='RepeatDataset',
#     times=1,
#     dataset=dict(
#         type='DdadDepthDataset',
#         data_root='/hdd_data/UrbanDepthData',
#         split='DDAD_train.txt',
#         depth_scale=1,
#         pipeline=train_depth_pipeline,
#         min_depth=1e-3,
#         max_depth=65
#     )
# )

dataset_probot_depth_val = dict(
    type='RepeatDataset',
    times=1,
    dataset=dict(
        type='ProbotDepthDataset',
        data_root=depth_data_root,
        ann_file='DepthProbot_val.txt',
        pipeline=test_depth_pipeline,
        depth_scale=1000,
        min_depth=1e-2,
        max_depth=65
    )
)

### SEGMENTATION ###

dataset_probot_seg_train = dict(
    type='RepeatDataset',
    times=1,
    dataset=dict(
            type=dataset_type,
            data_root=data_root,
            data_prefix = dict(img_path='Images', seg_map_path='Labels'),
            ann_file = 'train.txt',
            pipeline=train_seg_pipeline)
)

dataset_probot_seg_val = dict(
    type='RepeatDataset',
    times=1,
    dataset=dict(
            type=dataset_type,
            data_root=data_root,
            data_prefix = dict(img_path='Images', seg_map_path='Labels'),
            ann_file = 'val.txt',
            pipeline=test_pipeline)
)

#######       DATALOADERS       #######
train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            dataset_probot_depth_train,
            dataset_airsim_depth_train,
            # dataset_ddad_depth_train,
            dataset_probot_seg_train
    ]),
    sampler=dict(type='DefaultSampler', shuffle=True)  # necessary
)

val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    dataset=[
        dataset_probot_depth_val,
        dataset_probot_seg_val,
    ],
    sampler=dict(type='DefaultSampler', shuffle=False)  # necessary
)

test_dataloader = val_dataloader
