auto_scale_lr = dict(base_batch_size=48, enable=False)
backend_args = None
class_names = [
    'Pedestrian',
    'Cyclist',
    'Car',
]
point_cloud_range = [0, -25.6, -3, 51.2, 25.6, 2]
voxel_size = [0.16, 0.16, 5]
data_root = 'data/talk2radar/lidar/'
dataset_type = 'Talk2RadarDataset'
# db_sampler = dict(
#     backend_args=None,
#     classes=[
#         'Pedestrian',
#         'Cyclist',
#         'Car',
#     ],
#     data_root='data/kitti/',
#     info_path='data/kitti/kitti_dbinfos_train.pkl',
#     points_loader=dict(
#         backend_args=None,
#         coord_type='LIDAR',
#         load_dim=4,
#         type='LoadPointsFromFile',
#         use_dim=4),
#     prepare=dict(
#         filter_by_difficulty=[
#             -1,
#         ],
#         filter_by_min_points=dict(Car=5, Cyclist=5, Pedestrian=5)),
#     rate=1.0,
#     sample_groups=dict(Car=15, Cyclist=15, Pedestrian=15))
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='Det3DVisualizationHook'))
default_scope = 'mmdet3d'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))

eval_pipeline = [
    dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=4,
        type='LoadPointsFromFile',
        use_dim=4),
    dict(keys=[
        'points',
    ], type='Pack3DDetInputs'),
]
input_modality = dict(use_camera=False, use_lidar=True)
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
# lr = 0.001
metainfo = dict(classes=class_names)

model = dict(
    type='VoteNet',
    data_preprocessor=dict(type='Det3DDataPreprocessor'),
    backbone=dict(
        type='PointNet2SASSG',
        in_channels=4,
        num_points=(2048, 1024, 512, 256),
        radius=(0.2, 0.4, 0.8, 1.2),
        num_samples=(64, 32, 16, 16),
        sa_channels=((64, 64, 128), (128, 128, 256), (128, 128, 256),
                     (128, 128, 256)),
        fp_channels=((256, 256), (256, 256)),
        norm_cfg=dict(type='BN2d'),
        sa_cfg=dict(
            type='PointSAModule',
            pool_mod='max',
            use_xyz=True,
            normalize_xyz=True)),
    bbox_head=dict(
        type='VoteHead',
        vote_module_cfg=dict(
            in_channels=256,
            vote_per_seed=1,
            gt_per_seed=3,
            conv_channels=(256, 256),
            conv_cfg=dict(type='Conv1d'),
            norm_cfg=dict(type='BN1d'),
            norm_feats=True,
            vote_loss=dict(
                type='ChamferDistance',
                mode='l1',
                reduction='none',
                loss_dst_weight=10.0)),
        vote_aggregation_cfg=dict(
            type='PointSAModule',
            num_point=256,
            radius=0.3,
            num_sample=16,
            mlp_channels=[256, 128, 128, 128],
            use_xyz=True,
            normalize_xyz=True),
        pred_layer_cfg=dict(
            in_channels=128, shared_conv_channels=(128, 128), bias=True),
        objectness_loss=dict(
            type='mmdet.CrossEntropyLoss',
            class_weight=[0.2, 0.8],
            reduction='sum',
            loss_weight=5.0),
        center_loss=dict(
            type='ChamferDistance',
            mode='l2',
            reduction='sum',
            loss_src_weight=10.0,
            loss_dst_weight=10.0),
        num_classes=3,
        bbox_coder=dict(
        type='PartialBinBasedBBoxCoder',
        num_sizes=3,
        num_dir_bins=12,
        with_rot=True,
        mean_sizes=[
            [0.8, 0.6, 1.73],
            [1.76, 0.6, 1.73],
            [3.9, 1.6, 1.56],
        ]),
        dir_class_loss=dict(
            type='mmdet.CrossEntropyLoss', reduction='sum', loss_weight=1.0),
        dir_res_loss=dict(
            type='mmdet.SmoothL1Loss', reduction='sum', loss_weight=10.0),
        size_class_loss=dict(
            type='mmdet.CrossEntropyLoss', reduction='sum', loss_weight=1.0),
        size_res_loss=dict(
            type='mmdet.SmoothL1Loss', reduction='sum',
            loss_weight=10.0 / 3.0),
        semantic_loss=dict(
            type='mmdet.CrossEntropyLoss', reduction='sum', loss_weight=1.0)),
    
    
    # model training and testing settings
    train_cfg=dict(
        pos_distance_thr=0.3, neg_distance_thr=0.6, sample_mode='vote'),
    test_cfg=dict(
        sample_mode='seed',
        nms_thr=0.25,
        score_thr=0.05,
        per_class_proposal=True))

optim_wrapper = dict(
    clip_grad=dict(max_norm=35, norm_type=2),
    optimizer=dict(
        betas=(0.95,0.99), lr=0.0015, type='AdamW', weight_decay=0.01),
    type='OptimWrapper')
param_scheduler = [
    dict(
        T_max=32.0,
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=32.0,
        eta_min=0.01,
        type='CosineAnnealingLR'),
    dict(
        T_max=48.0,
        begin=32.0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=80,
        eta_min=1.0000000000000001e-07,
        type='CosineAnnealingLR'),
    dict(
        T_max=32.0,
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=32.0,
        eta_min=0.8947368421052632,
        type='CosineAnnealingMomentum'),
    dict(
        T_max=48.0,
        begin=32.0,
        convert_to_iter_based=True,
        end=80,
        eta_min=1,
        type='CosineAnnealingMomentum'),
]

resume = False

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=80,
    val_interval=2)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop') 
# test_cfg = dict()

train_pipeline = [
    dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=4,
        type='LoadPointsFromFile',
        use_dim=4),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(flip_ratio_bev_horizontal=0.5, type='RandomFlip3D'),
    dict(
        point_cloud_range=point_cloud_range,
        type='PointsRangeFilter'),
    dict(
        point_cloud_range=point_cloud_range,
        type='ObjectRangeFilter'),
    dict(type='PointShuffle'),
    dict(type='PointSample', num_points=20000),
    dict(
        keys=['points', 'gt_labels_3d','gt_bboxes_3d'],
        type='Pack3DDetInputs'),
]

test_pipeline = [
    dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=4,
        type='LoadPointsFromFile',
        use_dim=4),
    dict(type='RandomFlip3D'),
    dict(
        type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointSample', num_points=20000),
    dict(keys=['points'], type='Pack3DDetInputs'),
]

train_dataloader = dict(
    batch_size=4,
    dataset=dict(
            ann_file='talk2radar_infos_train.pkl',
            backend_args=None,
            box_type_3d='LiDAR',
            data_prefix=dict(pts='training/velodyne'),
            data_root=data_root,
            metainfo=metainfo,
            modality=input_modality,
            pipeline=train_pipeline,
            test_mode=False,
            type=dataset_type),
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))

test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        # ann_file='kitti_infos_val.pkl',
        ann_file='talk2radar_infos_val.pkl',
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

# test_evaluator = dict(
#     ann_file='data/vod/radar_5frames/kitti_infos_val.pkl',
#     backend_args=None,
#     metric='bbox',
#     type='KittiMetric') # for eval locally
test_evaluator = dict(
    type='KittiMetric',
    ann_file='data/talk2radar/lidar/talk2radar_infos_val.pkl',
    metric='bbox',
    format_only=True,
    submission_prefix='results/kitti-3class/kitti_results') # for submission

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
        ann_file='talk2radar_infos_val.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR'))

val_evaluator = dict(
    ann_file='data/talk2radar/lidar/talk2radar_infos_val.pkl',
    backend_args=None,
    metric='bbox',
    type='KittiMetric')

vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='Det3DLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
