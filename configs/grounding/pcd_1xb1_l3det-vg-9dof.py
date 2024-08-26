_base_ = ['../default_runtime.py']
n_points = 100000

backend_args = None
# Uncomment the following if use ceph or other file clients.
# See https://mmcv.readthedocs.io/en/latest/api.html#mmcv.fileio.FileClient
# for more details.
# file_client_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/scannet/':
#         's3://openmmlab/datasets/detection3d/scannet_processed/',
#         'data/scannet/':
#         's3://openmmlab/datasets/detection3d/scannet_processed/'
#     }))

metainfo = dict(classes='all')

model = dict(
    type='L3Det',
    data_preprocessor=dict(type='Det3DDataPreprocessor'),
    backbone=dict(
        type='PointNet2SASSG',
        in_channels=6,
        num_points=(2048, 1024, 512, 256),
        radius=(0.2, 0.4, 0.8, 1.2),
        num_samples=(64, 32, 16, 16),
        sa_channels=((64, 64, 128), (128, 128, 256), (128, 128, 256),
                     (128, 128, 256)),
        fp_channels=((256, 256), (256, 288)),
        norm_cfg=dict(type='BN2d'),
        sa_cfg=dict(
            type='PointSAModule',
            pool_mod='max',
            use_xyz=True,
            normalize_xyz=True)),
    bbox_head=dict(
        type='L3DetHead',
        in_channels=288,
        num_classes=256,
        size_cls_agnostic=False,
        num_decoder_layers=6,
        self_position_embedding='loc_learned',
        num_proposal=256,
        sampling_objectness_loss=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=8.0),
        objectness_loss=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        center_loss=dict(
            type='mmdet.SmoothL1Loss', reduction='sum', loss_weight=10.0),
        dir_class_loss=dict(
            type='mmdet.CrossEntropyLoss', reduction='sum', loss_weight=1.0),
        dir_res_loss=dict(
            type='mmdet.SmoothL1Loss', reduction='sum', loss_weight=10.0),
        size_class_loss=dict(
            type='mmdet.CrossEntropyLoss', reduction='sum', loss_weight=1.0),
        size_res_loss=dict(
            type='mmdet.SmoothL1Loss',
            beta=1.0,
            reduction='sum',
            loss_weight=10.0),
        semantic_loss=dict(
            type='mmdet.CrossEntropyLoss', reduction='sum', loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(sample_mode='kps'),
    test_cfg=dict(
        sample_mode='kps',
        nms_thr=0.25,
        score_thr=0.0,
        per_class_proposal=True,
        prediction_stages='last'))

dataset_type = 'PointCloud3DGroundingDataset'
data_root = 'data'

train_pipeline = [
    dict(type='LoadAnnotations3D'),
    dict(type='PointCloudPipeline'),
    dict(type='PointSample', num_points=n_points),
    dict(type='GlobalRotScaleTrans',
         rot_range=[-0.087266, 0.087266],
         scale_ratio_range=[.9, 1.1],
         translation_std=[.1, .1, .1],
         shift_height=False),
    dict(type='Pack3DDetInputs',
         keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(type='LoadAnnotations3D'),
    dict(type='PointCloudPipeline'),
    dict(type='PointSample', num_points=n_points),
    dict(type='Pack3DDetInputs',
         keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

# TODO: to determine a reasonable batch size
train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(type='RepeatDataset',
                 times=1,
                 dataset=dict(type=dataset_type,
                              data_root=data_root,
                              ann_file='embodiedscan_infos_train.pkl',
                              vg_file='embodiedscan_train_mini_vg.json',
                              metainfo=metainfo,
                              pipeline=train_pipeline,
                              test_mode=False,
                              filter_empty_gt=True,
                              box_type_3d='Euler-Depth')))

val_dataloader = dict(batch_size=1,
                      num_workers=1,
                      persistent_workers=True,
                      drop_last=False,
                      sampler=dict(type='DefaultSampler', shuffle=False),
                      dataset=dict(type=dataset_type,
                                   data_root=data_root,
                                   ann_file='embodiedscan_infos_val.pkl',
                                   vg_file='embodiedscan_val_mini_vg.json',
                                   metainfo=metainfo,
                                   pipeline=test_pipeline,
                                   test_mode=True,
                                   filter_empty_gt=True,
                                   box_type_3d='Euler-Depth'))
test_dataloader = val_dataloader

val_evaluator = dict(type='GroundingMetric')
test_evaluator = val_evaluator

# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=3)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# optimizer
lr = 0.006
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.0005),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'bbox_head.decoder_layers': dict(lr_mult=0.1, decay_mult=1.0),
            'bbox_head.decoder_self_posembeds': dict(
                lr_mult=0.1, decay_mult=1.0),
            'bbox_head.decoder_cross_posembeds': dict(
                lr_mult=0.1, decay_mult=1.0),
            'bbox_head.decoder_query_proj': dict(lr_mult=0.1, decay_mult=1.0),
            'bbox_head.decoder_key_proj': dict(lr_mult=0.1, decay_mult=1.0)
        }))

# learning rate
# learning rate
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=80,
        by_epoch=True,
        milestones=[56, 68],
        gamma=0.1)
]

custom_hooks = [dict(type='EmptyCacheHook', after_iter=True)]

# hooks
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3))

# vis_backends = [
#     dict(type='TensorboardVisBackend'),
#     dict(type='LocalVisBackend')
# ]
# visualizer = dict(
#     type='Det3DLocalVisualizer',
#     vis_backends=vis_backends, name='visualizer')

find_unused_parameters = True
load_from = '/mnt/petrelfs/lvruiyuan/repos/EmbodiedScan/work_dirs/pcd-3ddet/pcd-l3det-grounding.pth'  # noqa
