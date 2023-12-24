_base_ = [
    '../_base_/models/segformer_mit-b0.py',
    # '../_base_/datasets/probot_seg_depth_mmseg2.py',
    '../_base_/default_runtime.py'
]

# dataset settings

dataset_type = 'RoadsDataset'
data_root = '/hdd_data/asaf/12cm_train_roads/AllData_split/'
crop_size = (384, 384)

#######       DATA PIPELINES       #######

train_seg_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
       type='RandomResize',
       scale=(512, 512),
       ratio_range=(0.75, 2.0),
       keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    # dict(type='CenterCrop', crop_size=(512, 512)),
    # dict(type='Resize', scale=(1280, 720), keep_ratio=True),
    dict(type='PackSegInputs')
]

#######       DATASETS       #######

### TRAIN ###

dataset_train = dict(
    type='RepeatDataset',
    times=1,
    dataset=dict(
            type=dataset_type,
            data_root=data_root,
            data_prefix = dict(img_path='Images', seg_map_path='Labels'),
            ann_file = 'train.txt',
            pipeline=train_seg_pipeline)
)
### VALIDATION ###

dataset_val = dict(
    type='RepeatDataset',
    times=1,
    dataset=dict(
            type=dataset_type,
            data_root=data_root,
            data_prefix = dict(img_path='Images', seg_map_path='Labels'),
            ann_file = 'val.txt',
            pipeline=test_pipeline)
)

#######       TRAIN DATALOADERS       #######

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    dataset=dict(
        type='RepeatDataset',
        times=1,
        dataset=dataset_train,
    ),
    sampler=dict(type='InfiniteSampler', shuffle=True)  # necessary
)

#######       VAL DATALOADERS       #######

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    dataset=dict(
        type='RepeatDataset',
        times=1,
        dataset=dataset_val,
    ),
    sampler=dict(type='DefaultSampler', shuffle=False)  # necessary
)

test_dataloader = val_dataloader

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
find_unused_parameters = True

model = dict(
    type='EncoderDecoder',
    data_preprocessor = dict(
        type='SegDataPreProcessor',
        mean=[104.00699, 116.66877, 122.67892],
        std=[57.375, 57.375, 57.375],
        size=crop_size,
        bgr_to_rgb=False,
        pad_val=0,
        seg_pad_val=255),
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='pretrain/mit_b4_mmseg2.pth'),
        embed_dims=64,
        num_layers=[3, 8, 27, 3]),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, loss_name='loss_Roads'),
        ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=2e-4, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)
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
        end=420000,
        by_epoch=False,
    )
]

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

# The `val_interval` is the original `evaluation.interval`.
train_cfg = dict(type='IterBasedTrainLoop', max_iters=420000, val_interval=4000)
val_cfg = dict(type='ValLoop') # Use the default validation loop.
test_cfg = val_cfg # Use the default test loop.

default_hooks = dict(
    # record the time of every iterations.
    timer=dict(type='IterTimerHook'),

    # print log every 50 iterations.
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),

    # enable the parameter scheduler.
    param_scheduler=dict(type='ParamSchedulerHook'),

    # save checkpoint every 2000 iterations.
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=4000),

    # set sampler seed in distributed environment.
    sampler_seed=dict(type='DistSamplerSeedHook'),

    # validation results visualization.
    visualization=dict(type='SegVisualizationHook', draw=True))

vis_backends = [dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')

work_dir = '/home/iariav/Deep/Pytorch/mmsegmentation/work_dirs/Materials/b4_roads'