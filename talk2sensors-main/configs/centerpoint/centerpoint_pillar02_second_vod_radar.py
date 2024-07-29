auto_scale_lr = dict(base_batch_size=48, enable=False)
backend_args = None
class_names = [
    'Pedestrian',
    'Cyclist',
    'Car',
]
point_cloud_range = [0, -25.6, -3, 51.2, 25.6, 2]
voxel_size = [0.16, 0.16, 5]
data_root = 'data/vod/radar_5frames/'
dataset_type = 'KittiDataset'
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
        load_dim=7,
        type='LoadPointsFromFile',
        use_dim=7),
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
    type='CenterPoint',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=10,
            max_voxels=(16000, 40000),
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size)),
    # pts_voxel_encoder=dict(
    #     type='PillarFeatureNet',
    #     in_channels=5,
    #     feat_channels=[64],
    #     with_distance=False,
    #     voxel_size=(0.2, 0.2, 8),
    #     norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
    #     legacy=False),
    pts_voxel_encoder=dict(
        feat_channels=[64],
        in_channels=0,
        point_cloud_range=point_cloud_range,
        type='Radar7PillarFeatureNet',
        voxel_size=voxel_size,
        with_distance=False),
    pts_middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=(320, 320)),
    pts_backbone=dict(
        type='SECOND',
        in_channels=64,
        out_channels=[64, 128, 256],
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        out_channels=[128, 128, 128],
        upsample_strides=[0.5, 1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    pts_bbox_head=dict(
        type='CenterHead',
        in_channels=sum([128, 128, 128]),
        # tasks=[
        #     dict(num_class=1, class_names=['car']),
        #     dict(num_class=2, class_names=['truck', 'construction_vehicle']),
        #     dict(num_class=2, class_names=['bus', 'trailer']),
        #     dict(num_class=1, class_names=['barrier']),
        #     dict(num_class=2, class_names=['motorcycle', 'bicycle']),
        #     dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
        # ],
        tasks=[
            dict(num_class=1, class_names=['Car']),
            dict(num_class=1, class_names=['Cyclist']),
            dict(num_class=1, class_names=['Pedestrian']),
        ],
        common_heads=dict(
            # reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2)), # by zrx
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            # post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            post_center_range=[0, -25.6, -3, 51.2, 25.6, 2],
            pc_range = [0, -25.6], # added by zrx
            max_num=500,
            score_threshold=0.1,
            out_size_factor=4,
            voxel_size=voxel_size[:2],
            # code_size=9),
            code_size=7), # by zrx
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='mmdet.GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(
            type='mmdet.L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            point_cloud_range=point_cloud_range, #added by zrx
            # grid_size=[512, 512, 1],
            grid_size=[320, 320, 1], # by zrx
            voxel_size=voxel_size,
            out_size_factor=4,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            # code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2])),
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])), # by zrx
    test_cfg=dict(
        pts=dict(
            post_center_limit_range=[0, -25.6, -3, 51.2, 25.6, 2],
            max_per_img=500,
            max_pool_nms=False,
            # min_radius=[4, 12, 10, 1, 0.85, 0.175],
            min_radius=[4, 0.85, 0.175], # car, cyclist, pedestrian by zrx
            score_threshold=0.1,
            pc_range=[0, -25.6], # need to change
            out_size_factor=4,
            voxel_size=voxel_size[:2],
            nms_type='rotate',
            pre_max_size=1000,
            post_max_size=83,
            nms_thr=0.2)))

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
    val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop') 
# test_cfg = dict()

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

train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        dataset=dict(
            ann_file='kitti_infos_train.pkl',
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

test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='kitti_infos_val.pkl',
        # ann_file='kitti_infos_test.pkl',
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
    # ann_file='data/vod/radar_5frames/kitti_infos_test.pkl',
    ann_file='data/vod/radar_5frames/kitti_infos_val.pkl',
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
        ann_file='kitti_infos_val.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR'))

val_evaluator = dict(
    ann_file='data/vod/radar_5frames/kitti_infos_val.pkl',
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
