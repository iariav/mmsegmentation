_base_ = [
    '../_base_/models/segformer_mit-b0.py',
    # '../_base_/datasets/probot_seg_depth_mmseg2.py',
    '../_base_/default_runtime.py'
]

# dataset settings

dataset_type = 'ProbotDataset'
data_root = '/ssd_data/NAS/thermal/'
crop_size = (504, 504)

#######       DATA PIPELINES       #######
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
    # dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        color_type='unchanged',
        imdecode_backend='cv2'),
    dict(type='Resize', scale=(640, 512), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=16,
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
    sampler=dict(type='InfiniteSampler', shuffle=True)  # necessary
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


# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
# norm_cfg = dict(type='BN', requires_grad=True)
find_unused_parameters = True

resume = False
# load_from = '/home/iariav/Deep/Pytorch/mmsegmentation/work_dirs/b4_sgd_test_seg/iter_160000.pth'

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
        init_cfg=dict(type='Pretrained', checkpoint='pretrain/mit_b4_mmseg2.pth'),
        embed_dims=64,
        in_channels=1,
        num_layers=[3, 8, 27, 3]),
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
    # auxiliary_head=dict(
    #     type='DPTHeadDepth',
    #     in_channels=(64, 128, 320, 512),
    #     channels=256,
    #     embed_dims=768,
    #     post_process_channels=[64, 128, 320, 512],
    #     num_classes=150,
    #     readout_type='ignore',
    #     input_transform='multiple_select',
    #     in_index=(0, 1, 2, 3),
    #     norm_cfg=norm_cfg,
    #     loss_decode=[
    #     # dict(type='SupervisedDepthLoss', num_scales=1, loss_type='Sparse-L1',loss_weight=1.0),
    #     dict(type='SupervisedDepthLoss', num_scales=1, loss_type='Sparse-mse',loss_weight=1.0)
    #     ]),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# optimizer
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 # 'backbone': dict(lr_mult=1.0),
                                                 'head': dict(lr_mult=10.)
                                                 # 'auxiliary_head': dict(lr_mult=1.)
                                                 }),
    clip_grad=dict(max_norm=1, norm_type=2))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-7, by_epoch=False, begin=0, end=4000),
    dict(
        type='PolyLR',
        eta_min=1e-8,
        power=0.9,
        begin=4000,
        end=320000,
        by_epoch=False,
    )
]

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

# The `val_interval` is the original `evaluation.interval`.
train_cfg = dict(type='IterBasedTrainLoop', max_iters=320000, val_interval=2000)
val_cfg = val_cfg = dict(type='ValLoop') # Use the default validation loop.
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
    visualization=dict(type='SegVisualizationHook', draw=True, interval=100))

vis_backends = [dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')

work_dir = '/home/iariav/Deep/Pytorch/mmsegmentation/work_dirs/b4_thermal'