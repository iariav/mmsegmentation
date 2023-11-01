_base_ = [
    '../_base_/models/segformer_mit-b0.py',
    '../_base_/default_runtime.py'
]

# dataset settings
max_depth = 65.0
dataset_type = 'ProbotDataset'
# dataset_type = 'ProbotDatasetClear'
data_root = '/ssd_data/NAS/thermal/'
depth_data_root = data_root
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (416, 512)
test_crop_size = (512,640)

#######       DATA PIPELINES       #######
train_seg_pipeline = [
    dict(
        type='LoadImageFromFile',
        color_type='unchanged',
        imdecode_backend='cv2'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(640, 512),
        ratio_range=(0.85, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

train_depth_pipeline = [
    dict(
        type='LoadImageFromFile',
        color_type='unchanged',
        imdecode_backend='cv2'),
    dict(type='DepthLoadAnnotationsNpy'),
    dict(
        type='RandomResize',
        scale=(640, 512),
        ratio_range=(0.85, 1.2),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomFillDepthData', max_depth=max_depth),
    dict(type='PackDepthInputs')
]

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        color_type='unchanged',
        imdecode_backend='cv2'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=(640, 512), keep_ratio=True),
    # dict(type='CenterCrop', crop_size=test_crop_size),
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
    dict(
        type='LoadImageFromFile',
        color_type='unchanged',
        imdecode_backend='cv2'),
    dict(type='DepthLoadAnnotationsNpy'),
    dict(type='Resize', scale=(640, 512), keep_ratio=True),
    # dict(type='CenterCrop', crop_size=test_crop_size),
    dict(type='PackDepthInputs')
]

#######       DATASETS       #######

### DEPTH ###
dataset_probot_depth_train = dict(
    type='RepeatDataset',
    times=1,
    dataset=dict(
        type='ProbotDepthDataset',
        data_root=depth_data_root,
        ann_file='train_depth.txt',
        pipeline=train_depth_pipeline,
        depth_scale=1000,
        min_depth=1e-2,
        max_depth=max_depth
    )
)

# dataset_airsim_depth_train = dict(
#     type='RepeatDataset',
#     times=1,
#     dataset=dict(
#         type='ProbotDepthDataset',
#         data_root=depth_data_root,
#         ann_file='DepthProbot_Airsim_train.txt',
#         pipeline=train_depth_pipeline,
#         depth_scale=1,
#         min_depth=1e-2,
#         max_depth=65
#     )
# )

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
        ann_file='val_depth.txt',
        pipeline=test_depth_pipeline,
        depth_scale=1000,
        min_depth=1e-2,
        max_depth=max_depth
    )
)

### SEGMENTATION ###

dataset_probot_seg_train = dict(
    type='RepeatDataset',
    times=1,
    dataset=dict(
            type=dataset_type,
            data_root=data_root,
            data_prefix = dict(img_path='ThermalImages', seg_map_path='ThermalLabels'),
            ann_file = 'trainval.txt',
            pipeline=train_seg_pipeline)
)

dataset_probot_seg_val = dict(
    type='RepeatDataset',
    times=1,
    dataset=dict(
            type=dataset_type,
            data_root=data_root,
            data_prefix = dict(img_path='ThermalImages', seg_map_path='ThermalLabels'),
            ann_file = 'val.txt',
            pipeline=test_pipeline)
)

#######       DATALOADERS       #######
train_depth_dataloader = dict(
    batch_size=16,
    num_workers=8,
    dataset=dict(
        type='RepeatDataset',
        times=1,
        dataset=dataset_probot_depth_train,
    ),
    sampler=dict(type='InfiniteSampler', shuffle=True)  # necessary
)

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    dataset=dict(
        type='RepeatDataset',
        times=1,
        dataset=dataset_probot_seg_train,
    ),
    sampler=dict(type='InfiniteSampler', shuffle=True)  # necessary
)


val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type='RepeatDataset',
        times=1,
        dataset=dataset_probot_seg_val,
    ),
    sampler=dict(type='DefaultSampler', shuffle=False)  # necessary
)

val_depth_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type='RepeatDataset',
        times=1,
        dataset=dataset_probot_depth_val,
    ),
    sampler=dict(type='DefaultSampler', shuffle=False)  # necessary
)

test_dataloader = val_dataloader


# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
# norm_cfg = dict(type='BN', requires_grad=True)
find_unused_parameters = True

# load_from = '/home/iariav/Deep/Pytorch/mmsegmentation/work_dirs/b4_sgd_pixelformer/iter_92000.pth'

model = dict(
    type='EncoderDecoder',
    data_preprocessor = dict(
        type='SegDataPreProcessor',
        mean=[122.5],
        std=[57.375],
        size=crop_size,
        bgr_to_rgb=False,
        pad_val=0,
        seg_pad_val=255),
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='pretrain/mit_b3_mmseg2.pth'),
        in_channels=1,
        embed_dims=64,
        num_layers=[3, 4, 18, 3]),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='PixelFormerHead',
        num_heads=[4, 8, 16, 32],
        in_channels=[64, 128, 320, 512],
        channels=256,
        embed_dims=512,
        # depths=[2, 2, 18, 2],
        num_classes=150,
        post_process_channels=[64, 128, 320, 512],
        input_transform='multiple_select',
        in_index=(0, 1, 2, 3),
        norm_cfg=norm_cfg,
        window_size=7,
        max_depth=max_depth,
        min_depth=1e-2,
        pretrained='/home/iariav/Deep/Pytorch/PixelFormer/pretrained/swinv2_base_patch4_window12_192_22k_384.pth',
        loss_decode=[
            dict(type='SupervisedDepthLoss', num_scales=1, loss_type='Sparse-L1',loss_weight=0.5),
            dict(type='SupervisedDepthLoss', num_scales=1, loss_type='Sparse-silog',loss_weight=0.5)
            # dict(type='DepthSmoothnessLoss', num_scales=1, loss_weight=10.0)
        ]),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=2e-4, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 # 'backbone': dict(lr_mult=0.1),
                                                 # 'decode_head': dict(lr_mult=10.),
                                                 # 'auxiliary_head': dict(lr_mult=10.)
                                                 }),
    clip_grad=dict(max_norm=1, norm_type=2))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-8, by_epoch=False, begin=0, end=8000),
    dict(
        type='PolyLR',
        eta_min=1e-8,
        power=0.9,
        begin=8000,
        end=360000,
        by_epoch=False,
    )
]

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
val_depth_evaluator = dict(type='DepthMetric', min_depth_eval=0.1, max_depth_eval=max_depth)#, depth_metrics=['RMSE'])
test_evaluator = val_evaluator

# The `val_interval` is the original `evaluation.interval`.
train_cfg = dict(type='TwoDataloadersIterBasedTrainLoop', DepthDataloader=train_depth_dataloader, max_iters=360000, val_interval=4000
                 )
val_cfg = dict(type='TwoDataloadersValLoop', DepthDataloader= val_depth_dataloader, DepthEvaluator=val_depth_evaluator)
test_cfg = val_cfg # Use the default test loop.

default_hooks = dict(
    # record the time of every iterations.
    timer=dict(type='IterTimerHook'),

    # print log every 50 iterations.
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),

    # enable the parameter scheduler.
    param_scheduler=dict(type='ParamSchedulerHook'),

    # save checkpoint every 2000 iterations.
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=8000),

    # set sampler seed in distributed environment.
    sampler_seed=dict(type='DistSamplerSeedHook'),

    # validation results visualization.
    visualization=dict(type='SegVisualizationHook', draw=True))

vis_backends = [dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')

work_dir = '/home/iariav/Deep/Pytorch/mmsegmentation/work_dirs/b3_sgd_pixelformer_thermal_with_ddad'