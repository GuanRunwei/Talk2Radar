auto_scale_lr = dict(base_batch_size=48, enable=False)
backend_args = None
class_names = [
    'Pedestrian',
    'Cyclist',
    'Car',
]
data_root = 'data/vod/lidar/'
dataset_type = 'KittiDataset'
# db_sampler = dict(
#     backend_args=None,
#     classes=[
#         'Pedestrian',
#         'Cyclist',
#         'Car',
#     ],
#     data_root='data/vod/lidar/',
#     info_path='data/vod/lidar/kitti_dbinfos_train.pkl',
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
    checkpoint=dict(interval=-1, type='CheckpointHook'),
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
epoch_num = 80
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
lr = 0.001
metainfo = dict(classes=[
    'Pedestrian',
    'Cyclist',
    'Car',
])
model = dict(
    backbone=dict(
        in_channels=64,
        layer_nums=[
            3,
            5,
            5,
        ],
        layer_strides=[
            2,
            2,
            2,
        ],
        out_channels=[
            64,
            128,
            256,
        ],
        type='SECOND'),
    bbox_head=dict(
        anchor_generator=dict(
            ranges=[
                [
                    0,
                    -39.68,
                    -0.6,
                    69.12,
                    39.68,
                    -0.6,
                ],
                [
                    0,
                    -39.68,
                    -0.6,
                    69.12,
                    39.68,
                    -0.6,
                ],
                [
                    0,
                    -39.68,
                    -1.78,
                    69.12,
                    39.68,
                    -1.78,
                ],
            ],
            reshape_out=False,
            rotations=[
                0,
                1.57,
            ],
            sizes=[
                [
                    0.8,
                    0.6,
                    1.73,
                ],
                [
                    1.76,
                    0.6,
                    1.73,
                ],
                [
                    3.9,
                    1.6,
                    1.56,
                ],
            ],
            type='AlignedAnchor3DRangeGenerator'),
        assign_per_class=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        diff_rad_by_sin=True,
        feat_channels=384,
        in_channels=384,
        loss_bbox=dict(
            beta=0.1111111111111111,
            loss_weight=2.0,
            type='mmdet.SmoothL1Loss'),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='mmdet.FocalLoss',
            use_sigmoid=True),
        loss_dir=dict(
            loss_weight=0.2, type='mmdet.CrossEntropyLoss', use_sigmoid=False),
        num_classes=3,
        type='Anchor3DHead',
        use_direction_classifier=True),
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=32,
            max_voxels=(
                16000,
                40000,
            ),
            point_cloud_range=[
                0,
                -39.68,
                -3,
                69.12,
                39.68,
                1,
            ],
            voxel_size=[
                0.16,
                0.16,
                4,
            ])),
    middle_encoder=dict(
        in_channels=64, output_shape=[
            496,
            432,
        ], type='PointPillarsScatter'),
    neck=dict(
        in_channels=[
            64,
            128,
            256,
        ],
        out_channels=[
            128,
            128,
            128,
        ],
        type='SECONDFPN',
        upsample_strides=[
            1,
            2,
            4,
        ]),
    test_cfg=dict(
        max_num=50,
        min_bbox_size=0,
        nms_across_levels=False,
        nms_pre=100,
        nms_thr=0.01,
        score_thr=0.1,
        use_rotate_nms=True),
    train_cfg=dict(
        allowed_border=0,
        assigner=[
            dict(
                ignore_iof_thr=-1,
                iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                min_pos_iou=0.35,
                neg_iou_thr=0.35,
                pos_iou_thr=0.5,
                type='Max3DIoUAssigner'),
            dict(
                ignore_iof_thr=-1,
                iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                min_pos_iou=0.35,
                neg_iou_thr=0.35,
                pos_iou_thr=0.5,
                type='Max3DIoUAssigner'),
            dict(
                ignore_iof_thr=-1,
                iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                min_pos_iou=0.45,
                neg_iou_thr=0.45,
                pos_iou_thr=0.6,
                type='Max3DIoUAssigner'),
        ],
        debug=False,
        pos_weight=-1),
    type='VoxelNet',
    voxel_encoder=dict(
        feat_channels=[
            64,
        ],
        in_channels=4,
        point_cloud_range=[
            0,
            -39.68,
            -3,
            69.12,
            39.68,
            1,
        ],
        type='PillarFeatureNet',
        voxel_size=[
            0.16,
            0.16,
            4,
        ],
        with_distance=False))
optim_wrapper = dict(
    clip_grad=dict(max_norm=35, norm_type=2),
    optimizer=dict(
        betas=(
            0.95,
            0.99,
        ), lr=0.001, type='AdamW', weight_decay=0.01),
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
point_cloud_range = [
    0,
    -25.6,
    -3,
    51.2,
    25.6,
    2,
]
resume = False
test_cfg = dict()
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='kitti_infos_val.pkl',
        backend_args=None,
        box_type_3d='LiDAR',
        data_prefix=dict(pts='training/velodyne_reduced'),
        data_root='data/vod/lidar/',
        metainfo=dict(classes=[
            'Pedestrian',
            'Cyclist',
            'Car',
        ]),
        modality=dict(use_camera=False, use_lidar=True),
        pipeline=[
            dict(
                backend_args=None,
                coord_type='LIDAR',
                load_dim=4,
                type='LoadPointsFromFile',
                use_dim=4),
            # dict(
            #     flip=False,
            #     img_scale=(
            #         1333,
            #         800,
            #     ),
            #     pts_scale_ratio=1,
            #     transforms=[
            #         dict(
            #             rot_range=[
            #                 0,
            #                 0,
            #             ],
            #             scale_ratio_range=[
            #                 1.0,
            #                 1.0,
            #             ],
            #             translation_std=[
            #                 0,
            #                 0,
            #                 0,
            #             ],
            #             type='GlobalRotScaleTrans'),
            #         dict(type='RandomFlip3D'),
            #         dict(
            #             point_cloud_range=[
            #                 0,
            #                 -25.6,
            #                 -3,
            #                 51.2,
            #                 25.6,
            #                 2,
            #             ],
            #             type='PointsRangeFilter'),
            #     ],
            #     type='MultiScaleFlipAug3D'),
            dict(keys=[
                'points',
            ], type='Pack3DDetInputs'),
        ],
        test_mode=True,
        type='KittiDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='data/vod/lidar/kitti_infos_val.pkl',
    backend_args=None,
    metric='bbox',
    type='KittiMetric')
test_pipeline = [
    dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=4,
        type='LoadPointsFromFile',
        use_dim=4),
    dict(
        flip=False,
        img_scale=(
            1333,
            800,
        ),
        pts_scale_ratio=1,
        transforms=[
            dict(
                rot_range=[
                    0,
                    0,
                ],
                scale_ratio_range=[
                    1.0,
                    1.0,
                ],
                translation_std=[
                    0,
                    0,
                    0,
                ],
                type='GlobalRotScaleTrans'),
            dict(type='RandomFlip3D'),
            dict(
                point_cloud_range=[
                    0,
                    -25.6,
                    -3,
                    51.2,
                    25.6,
                    2,
                ],
                type='PointsRangeFilter'),
        ],
        type='MultiScaleFlipAug3D'),
    dict(keys=[
        'points',
    ], type='Pack3DDetInputs'),
]
train_cfg = dict(by_epoch=True, max_epochs=80, val_interval=2)
train_dataloader = dict(
    batch_size=6,
    dataset=dict(
        dataset=dict(
            ann_file='kitti_infos_train.pkl',
            backend_args=None,
            box_type_3d='LiDAR',
            data_prefix=dict(pts='training/velodyne_reduced'),
            data_root='data/vod/lidar/',
            metainfo=dict(classes=[
                'Pedestrian',
                'Cyclist',
                'Car',
            ]),
            modality=dict(use_camera=False, use_lidar=True),
            pipeline=[
                dict(
                    backend_args=None,
                    coord_type='LIDAR',
                    load_dim=4,
                    type='LoadPointsFromFile',
                    use_dim=4),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True),
                # dict(
                #     db_sampler=dict(
                #         backend_args=None,
                #         classes=[
                #             'Pedestrian',
                #             'Cyclist',
                #             'Car',
                #         ],
                #         data_root='data/vod/lidar/',
                #         info_path='data/vod/lidar/kitti_dbinfos_train.pkl',
                #         points_loader=dict(
                #             backend_args=None,
                #             coord_type='LIDAR',
                #             load_dim=4,
                #             type='LoadPointsFromFile',
                #             use_dim=4),
                #         prepare=dict(
                #             filter_by_difficulty=[
                #                 -1,
                #             ],
                #             filter_by_min_points=dict(
                #                 Car=5, Cyclist=5, Pedestrian=5)),
                #         rate=1.0,
                #         sample_groups=dict(Car=15, Cyclist=15, Pedestrian=15)),
                #     type='ObjectSample',
                #     use_ground_plane=False),
                # dict(flip_ratio_bev_horizontal=0.5, type='RandomFlip3D'),
                # dict(
                #     rot_range=[
                #         -0.78539816,
                #         0.78539816,
                #     ],
                #     scale_ratio_range=[
                #         0.95,
                #         1.05,
                #     ],
                #     type='GlobalRotScaleTrans'),
                dict(
                    point_cloud_range=[
                        0,
                        -25.6,
                        -3,
                        51.2,
                        25.6,
                        2,
                    ],
                    type='PointsRangeFilter'),
                dict(
                    point_cloud_range=[
                        0,
                        -25.6,
                        -3,
                        51.2,
                        25.6,
                        2,
                    ],
                    type='ObjectRangeFilter'),
                dict(type='PointShuffle'),
                dict(
                    keys=[
                        'points',
                        'gt_labels_3d',
                        'gt_bboxes_3d',
                    ],
                    type='Pack3DDetInputs'),
            ],
            test_mode=False,
            type='KittiDataset'),
        times=2,
        type='RepeatDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=4,
        type='LoadPointsFromFile',
        use_dim=4),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    # dict(
    #     db_sampler=dict(
    #         backend_args=None,
    #         classes=[
    #             'Pedestrian',
    #             'Cyclist',
    #             'Car',
    #         ],
    #         data_root='data/vod/lidar/',
    #         info_path='data/vod/lidar/kitti_dbinfos_train.pkl',
    #         points_loader=dict(
    #             backend_args=None,
    #             coord_type='LIDAR',
    #             load_dim=4,
    #             type='LoadPointsFromFile',
    #             use_dim=4),
    #         prepare=dict(
    #             filter_by_difficulty=[
    #                 -1,
    #             ],
    #             filter_by_min_points=dict(Car=5, Cyclist=5, Pedestrian=5)),
    #         rate=1.0,
    #         sample_groups=dict(Car=15, Cyclist=15, Pedestrian=15)),
    #     type='ObjectSample',
    #     use_ground_plane=False),
    # dict(flip_ratio_bev_horizontal=0.5, type='RandomFlip3D'),
    # dict(
    #     rot_range=[
    #         -0.78539816,
    #         0.78539816,
    #     ],
    #     scale_ratio_range=[
    #         0.95,
    #         1.05,
    #     ],
    #     type='GlobalRotScaleTrans'),
    dict(
        point_cloud_range=[
            0,
            -25.6,
            -3,
            51.2,
            25.6,
            2,
        ],
        type='PointsRangeFilter'),
    dict(
        point_cloud_range=[
            0,
            -25.6,
            -3,
            51.2,
            25.6,
            2,
        ],
        type='ObjectRangeFilter'),
    dict(type='PointShuffle'),
    dict(
        keys=[
            'points',
            'gt_labels_3d',
            'gt_bboxes_3d',
        ],
        type='Pack3DDetInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='kitti_infos_val.pkl',
        backend_args=None,
        box_type_3d='LiDAR',
        data_prefix=dict(pts='training/velodyne_reduced'),
        data_root='data/vod/lidar/',
        metainfo=dict(classes=[
            'Pedestrian',
            'Cyclist',
            'Car',
        ]),
        modality=dict(use_camera=False, use_lidar=True),
        pipeline=[
            dict(
                backend_args=None,
                coord_type='LIDAR',
                load_dim=4,
                type='LoadPointsFromFile',
                use_dim=4),
            dict(
                flip=False,
                img_scale=(
                    1333,
                    800,
                ),
                pts_scale_ratio=1,
                transforms=[
                    dict(
                        rot_range=[
                            0,
                            0,
                        ],
                        scale_ratio_range=[
                            1.0,
                            1.0,
                        ],
                        translation_std=[
                            0,
                            0,
                            0,
                        ],
                        type='GlobalRotScaleTrans'),
                    dict(type='RandomFlip3D'),
                    dict(
                        point_cloud_range=[
                            0,
                            -25.6,
                            -3,
                            51.2,
                            25.6,
                            2,
                        ],
                        type='PointsRangeFilter'),
                ],
                type='MultiScaleFlipAug3D'),
            dict(keys=[
                'points',
            ], type='Pack3DDetInputs'),
        ],
        test_mode=True,
        type='KittiDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='data/vod/lidar/kitti_infos_val.pkl',
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
voxel_size = [
    0.16,
    0.16,
    4,
]
work_dir = './work_dirs/pointpillars_hv_secfpn_8xb6-160e_vod-3d-3class-lidar'
