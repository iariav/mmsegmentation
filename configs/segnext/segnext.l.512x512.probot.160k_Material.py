_base_ = [
    # './segnext_mscan-t_1xb16-adamw-160k_ade20k-512x512.py',
    # '../_base_/datasets/probot_seg_depth_mmseg2.py',
    '../_base_/default_runtime.py'
]

# dataset settings

checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/mscan_l_20230227-cef260d4.pth'  # noqa
dataset_type = 'MaterialsDataset'
data_root = '/hdd_data/asaf/ido_test/AllData_split/'
crop_size = (448, 448)

#######       DATA PIPELINES       #######

train_seg_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
       type='RandomResize',
       scale=(512, 512),
       ratio_range=(0.875, 1.5),
       keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    # dict(type='RandomRotFlip', rotate_prob=0.0, flip_prob=0.5),
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

dataset_material_train = dict(
    type='RepeatDataset',
    times=1,
    dataset=dict(
            type='MaterialsDataset',
            data_root=data_root,
            data_prefix = dict(img_path='Images', seg_map_path='Labels'),
            ann_file = 'trainval_material.txt',
            pipeline=train_seg_pipeline)
)

dataset_semanticextended_train = dict(
    type='RepeatDataset',
    times=1,
    dataset=dict(
            type='SemanticExtendedDataset',
            data_root=data_root,
            data_prefix = dict(img_path='Images', seg_map_path='Labels'),
            ann_file = 'trainval_semantic_extended.txt',
            pipeline=train_seg_pipeline)
)

dataset_semantic_train = dict(
    type='RepeatDataset',
    times=1,
    dataset=dict(
            type='SemanticDataset',
            data_root=data_root,
            data_prefix = dict(img_path='Images', seg_map_path='Labels'),
            ann_file = 'trainval_semantic.txt',
            pipeline=train_seg_pipeline)
)

dataset_morphology_train = dict(
    type='RepeatDataset',
    times=1,
    dataset=dict(
            type='MorphologyDataset',
            data_root=data_root,
            data_prefix = dict(img_path='Images', seg_map_path='Labels'),
            ann_file = 'trainval_material_morphology.txt',
            pipeline=train_seg_pipeline)
)

### VALIDATION ###

dataset_semantic_val = dict(
    type='RepeatDataset',
    times=1,
    dataset=dict(
            type='SemanticDataset',
            data_root=data_root,
            data_prefix = dict(img_path='Images', seg_map_path='Labels'),
            ann_file = 'val_semantic.txt',
            pipeline=test_pipeline)
)

dataset_semanticextended_val = dict(
    type='RepeatDataset',
    times=1,
    dataset=dict(
            type='SemanticExtendedDataset',
            data_root=data_root,
            data_prefix = dict(img_path='Images', seg_map_path='Labels'),
            ann_file = 'val_semantic_extended.txt',
            pipeline=test_pipeline)
)

dataset_material_val = dict(
    type='RepeatDataset',
    times=1,
    dataset=dict(
            type='MaterialsDataset',
            data_root=data_root,
            data_prefix = dict(img_path='Images', seg_map_path='Labels'),
            ann_file = 'val_material.txt',
            pipeline=test_pipeline)
)

dataset_morphology_val = dict(
    type='RepeatDataset',
    times=1,
    dataset=dict(
            type='MorphologyDataset',
            data_root=data_root,
            data_prefix = dict(img_path='Images', seg_map_path='Labels'),
            ann_file = 'val_material_morphology.txt',
            pipeline=test_pipeline)
)

#######       TRAIN DATALOADERS       #######

train_dataloader_semantic = dict(
    batch_size=16,
    num_workers=8,
    dataset=dict(
        type='RepeatDataset',
        times=1,
        dataset=dataset_semantic_train,
    ),
    sampler=dict(type='InfiniteSampler', shuffle=True)  # necessary
)

train_dataloader_semanticextended = dict(
    batch_size=16,
    num_workers=8,
    dataset=dict(
        type='RepeatDataset',
        times=1,
        dataset=dataset_semanticextended_train,
    ),
    sampler=dict(type='InfiniteSampler', shuffle=True)  # necessary
)

train_dataloader_material = dict(
    batch_size=16,
    num_workers=8,
    dataset=dict(
        type='RepeatDataset',
        times=1,
        dataset=dataset_material_train,
    ),
    sampler=dict(type='InfiniteSampler', shuffle=True)  # necessary
)

train_dataloader_morphology = dict(
    batch_size=16,
    num_workers=8,
    dataset=dict(
        type='RepeatDataset',
        times=1,
        dataset=dataset_morphology_train,
    ),
    sampler=dict(type='InfiniteSampler', shuffle=True)  # necessary
)

#######       VAL DATALOADERS       #######

val_dataloader_semantic = dict(
    batch_size=1,
    num_workers=1,
    dataset=dict(
        type='RepeatDataset',
        times=1,
        dataset=dataset_semantic_val,
    ),
    sampler=dict(type='DefaultSampler', shuffle=False)  # necessary
)

val_dataloader_semanticextended = dict(
    batch_size=1,
    num_workers=1,
    dataset=dict(
        type='RepeatDataset',
        times=1,
        dataset=dataset_semanticextended_val,
    ),
    sampler=dict(type='DefaultSampler', shuffle=False)  # necessary
)

val_dataloader_material = dict(
    batch_size=1,
    num_workers=1,
    dataset=dict(
        type='RepeatDataset',
        times=1,
        dataset=dataset_material_val,
    ),
    sampler=dict(type='DefaultSampler', shuffle=False)  # necessary
)

val_dataloader_morphology = dict(
    batch_size=1,
    num_workers=1,
    dataset=dict(
        type='RepeatDataset',
        times=1,
        dataset=dataset_morphology_val,
    ),
    sampler=dict(type='DefaultSampler', shuffle=False)  # necessary
)

test_dataloader = val_dataloader_semantic


# model settings
ham_norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
norm_cfg = dict(type='SyncBN', requires_grad=True)
find_unused_parameters = True

model = dict(
    type='EncoderDecoder',
    data_preprocessor = dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        size=crop_size,
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255),
    backbone=dict(
        type='MSCAN',
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        embed_dims=[64, 128, 320, 512],
        mlp_ratios=[8, 8, 4, 4],
        drop_rate=0.0,
        drop_path_rate=0.3,
        depths=[3, 5, 27, 3],
        attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
        attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]],
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='BN', requires_grad=True)),
    decode_head=dict(
        type='LightHamMultiHead',
        in_channels=[128, 320, 512],
        in_index=[1, 2, 3],
        channels=1024,
        ham_channels=256,
        dropout_ratio=0.1,
        num_classes=[10, 12, 28, 40],
        norm_cfg=ham_norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.5, loss_name='loss_Semantic'),
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, loss_name='loss_SemanticExtended'),
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, loss_name='loss_Material'),
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, loss_name='loss_MaterialMorphology')
        ],
        ham_kwargs = dict(
            MD_S=1,
            MD_R=16,
            train_steps=6,
            eval_steps=7,
            inv_t=100,
            rand_init=True),
),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.)
                                                 }),
    clip_grad=dict(max_norm=1, norm_type=2))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-8, by_epoch=False, begin=0, end=8000),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=8000,
        end=420000,
        by_epoch=False,
    )
]

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
train_dataloader = train_dataloader_semantic
val_dataloader = val_dataloader_semanticextended
# The `val_interval` is the original `evaluation.interval`.
train_cfg = dict(type='MultiDataloadersIterBasedTrainLoop',
                 dataloaders=[train_dataloader_semantic,
                              train_dataloader_semanticextended,
                              train_dataloader_material,
                              train_dataloader_morphology
                 ],
                 max_iters=420000,
                 val_interval=4000
                 )
# val_cfg = dict(type='MultiDataloadersValLoop',
#                  dataloaders=[val_dataloader_semantic,
#                               val_dataloader_semanticextended,
#                               val_dataloader_material,
#                               val_dataloader_morphology
#                  ],
#                  evaluators = [
#                                val_evaluator,
#                                val_evaluator,
#                                val_evaluator,
#                                val_evaluator
#                  ])
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

work_dir = '/home/iariav/Deep/Pytorch/mmsegmentation/work_dirs/Materials/segnext_l_material'