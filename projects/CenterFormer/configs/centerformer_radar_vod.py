_base_ = ['../../../configs/_base_/default_runtime.py']
custom_imports = dict(
    imports=['projects.CenterFormer.centerformer'], allow_failed_imports=False)

# model settings
# Voxel size for voxel encoder
# Usually voxel size is changed consistently with the point cloud range
# If point cloud range is modified, do remember to change all related
# keys in the config.
voxel_size = [0.1, 0.1, 0.125]
point_cloud_range = [0, -25.6, -3, 51.2, 25.6, 2]
class_names = ['Car', 'Pedestrian', 'Cyclist']
tasks = [dict(num_class=3, class_names = [
    'Pedestrian',
    'Cyclist',
    'Car',
])]
metainfo = dict(classes=class_names)
input_modality = dict(use_lidar=True, use_camera=False)
backend_args = None

model = dict(
    type='CenterFormer',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_type='dynamic',
        voxel_layer=dict(
            max_num_points=-1,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(-1, -1))),
    voxel_encoder=dict(
        type='DynamicSimpleVFE',
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size),
    middle_encoder=dict(
        type='SparseEncoder',
        in_channels=7,
        sparse_shape=[41, 512, 512],
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='naiveSyncBN1d', eps=0.001, momentum=0.01),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128,
                                                                      128)),
        encoder_paddings=((1, 1, 1), (1, 1, 1), (1, 1, [0, 1, 1]), (1, 1)),
        block_type='basicblock'),
    backbone=dict(
        type='DeformableDecoderRPN',
        layer_nums=[5, 5, 1],
        ds_num_filters=[256, 256, 128],
        num_input_features=256,
        tasks=tasks,
        use_gt_training=True,
        corner=True,
        assign_label_window_size=1,
        obj_num=500,
        norm_cfg=dict(type='SyncBN', eps=1e-3, momentum=0.01),
        transformer_config=dict(
            depth=2,
            n_heads=6,
            dim_single_head=64,
            dim_ffn=256,
            dropout=0.3,
            out_attn=False,
            n_points=15,
        ),
    ),
    bbox_head=dict(
        type='CenterFormerBboxHead',
        in_channels=256,
        tasks=tasks,
        # dataset='waymo',  # where is the calling of this parameter? None
        weight=2,
        corner_loss=True,
        iou_loss=True,
        assign_label_window_size=1,
        norm_cfg=dict(type='SyncBN', eps=1e-3, momentum=0.01),
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        common_heads={  # sub-regression heads from CenterPoint
            'reg': (2, 2),     # (x, y): 2D location of objects in the BEV
            'height': (1, 2),  # height of objects to form 3D location with (x, y)
            'dim': (3, 2),     # 3D dimensions of objects
            'rot': (2, 2),     # yaw of objects indicated by sin(yaw) and cos(yaw), as the continuous regression targets
            'iou': (1, 2)      # predict iou introduced by CIA-SSD: facilitate the bbox convergence during training;
                               #                                    and as the confidence score of bbox prediction during testing.
        },  # (output_channel, num_conv)
    ),
    train_cfg=dict(
        # both SECOND and CenterFormer(SECOND's middle_backbone) set the height of sparse_shape to 41 (then it will be 2 after encoding)
        # hence the height size of voxel is changed to keep the height of voxel grid as 40
        grid_size=[512, 512, 40],
        voxel_size=voxel_size,
        out_size_factor=4,
        dense_reg=1,
        gaussian_overlap=0.1,
        point_cloud_range=point_cloud_range,
        max_objs=500,
        min_radius=2,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    test_cfg=dict(
        post_center_limit_range=point_cloud_range,
        nms=dict(
            use_rotate_nms=False,
            use_multi_class_nms=True,
            nms_pre_max_size=[1600, 1600, 800],
            nms_post_max_size=[200, 200, 100],
            nms_iou_threshold=[0.8, 0.55, 0.55],
        ),
        score_threshold=0.1,
        pc_range=[0, -25.6],   # x轴和y轴的起始坐标，[point_cloud_range[0], point_cloud_range[1]]
        out_size_factor=4,
        voxel_size=[0.1, 0.1],
        obj_num=1000,
    ))

data_root = 'data/vod/radar_5frames/'

# kitti train_pipeline from zrx
train_pipeline = [
    dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=7,
        type='LoadPointsFromFile',
        use_dim=7),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(flip_ratio_bev_horizontal=0.5, type='RandomFlip3D'),
    dict(
        point_cloud_range=point_cloud_range,
        type='PointsRangeFilter'),
    dict(
        point_cloud_range=point_cloud_range,
        type='ObjectRangeFilter'),
    dict(type='PointShuffle'),
    dict(
        keys=['points', 'gt_labels_3d','gt_bboxes_3d'],
        type='Pack3DDetInputs'),
]

# kitti test_pipeline from zrx
test_pipeline = [
    dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=7,
        type='LoadPointsFromFile',
        use_dim=7),
    dict(type='RandomFlip3D'),
    dict(
        type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(keys=['points'], type='Pack3DDetInputs'),
]

dataset_type = 'KittiDataset'

# kitti train_dataloader from zrx
train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        dataset=dict(
            ann_file='vod_infos_train.pkl',
            backend_args=None,
            box_type_3d='LiDAR',
            data_prefix=dict(pts='training/velodyne'),
            data_root='data/vod/radar_5frames/',
            metainfo=metainfo,
            modality=input_modality,
            pipeline=train_pipeline,
            test_mode=False,
            type=dataset_type),
        times=2,
        type='RepeatDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))

# kitti val_dataloader and test_dataloader from zrx
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        # ann_file='kitti_infos_val.pkl',
        ann_file='vod_infos_val.pkl',
        backend_args=None,
        box_type_3d='LiDAR',
        data_prefix=dict(pts='training/velodyne'),
        data_root=data_root,
        metainfo=metainfo,
        modality=input_modality,
        pipeline=test_pipeline,
        test_mode=True,
        type=dataset_type),
    drop_last=False,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts='training/velodyne'),
        ann_file='kitti_infos_val.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR'))

# original val_evaluator and test_evaluator of CenterFormer
# val_evaluator = dict(
#     type='WaymoMetric', waymo_bin_file='./data/waymo/waymo_format/gt.bin')
# test_evaluator = val_evaluator

# kitti val_evaluator and test_evaluator from zrx
test_evaluator = dict(
    type='KittiMetric',
    ann_file='data/vod/radar_5frames/kitti_infos_val.pkl',
    metric='bbox',
    format_only=True,
    submission_prefix='results/kitti-3class/kitti_results') # for submission
val_evaluator = dict(
    ann_file='data/vod/radar_5frames/kitti_infos_val.pkl',
    backend_args=None,
    metric='bbox',
    type='KittiMetric')

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# For waymo dataset, we usually evaluate the model at the end of training.
# Since the models are trained by 24 epochs by default, we set evaluation
# interval to be 20. Please change the interval accordingly if you do not
# use a default schedule.
# optimizer
lr = 3e-4
# This schedule is mainly used by models on nuScenes dataset
# max_norm=10 is better for SECOND
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01, betas=(0.9, 0.99)),
    clip_grad=dict(max_norm=35, norm_type=2))
# learning rate
param_scheduler = [
    # learning rate scheduler
    # During the first 8 epochs, learning rate increases from 0 to lr * 10
    # during the next 12 epochs, learning rate decreases from lr * 10 to
    # lr * 1e-4
    dict(
        type='CosineAnnealingLR',
        T_max=8,
        eta_min=lr * 10,
        begin=0,
        end=8,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=12,
        eta_min=lr * 1e-4,
        begin=8,
        end=20,
        by_epoch=True,
        convert_to_iter_based=True),
    # momentum scheduler
    # During the first 8 epochs, momentum increases from 0 to 0.85 / 0.95
    # during the next 12 epochs, momentum increases from 0.85 / 0.95 to 1
    dict(
        type='CosineAnnealingMomentum',
        T_max=8,
        eta_min=0.85 / 0.95,
        begin=0,
        end=8,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=12,
        eta_min=1,
        begin=8,
        end=20,
        by_epoch=True,
        convert_to_iter_based=True)
]

# runtime settings
train_cfg = dict(by_epoch=True, max_epochs=20, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (4 GPUs) x (4 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=5))
# custom_hooks = [dict(type='DisableObjectSampleHook', disable_after_epoch=15)]
