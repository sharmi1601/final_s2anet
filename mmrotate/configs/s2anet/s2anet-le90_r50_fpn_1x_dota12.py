# 12-class S2A-Net (le90) on DOTA v1.0
# novel classes removed: plane, baseball-diamond, tennis-court

_base_ = [
    '../_base_/datasets/dota.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
]

# --- safer on Windows to avoid loader issues
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

angle_version = 'le90'
CLASSES_12 = (
    'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle',
    'ship', 'basketball-court', 'storage-tank', 'soccer-ball-field',
    'roundabout', 'harbor', 'swimming-pool', 'helicopter'
)
metainfo = dict(classes=CLASSES_12)

data_root = 'data/base_training/'

model = dict(
    type='RefineSingleStageDetector',
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True, pad_size_divisor=32, boxtype2tensor=False),
    backbone=dict(
        type='mmdet.ResNet', depth=50, num_stages=4, out_indices=(0,1,2,3),
        frozen_stages=1, zero_init_residual=False,
        norm_cfg=dict(type='BN', requires_grad=True), norm_eval=True,
        style='pytorch', init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='mmdet.FPN', in_channels=[256,512,1024,2048], out_channels=256,
        start_level=1, add_extra_convs='on_input', num_outs=5),

    # ---- init head (stage 0)
    bbox_head_init=dict(
        type='S2AHead', num_classes=12, in_channels=256,
        stacked_convs=2, feat_channels=256,
        anchor_generator=dict(
            type='FakeRotatedAnchorGenerator', angle_version=angle_version,
            scales=[4], ratios=[1.0], strides=[8,16,32,64,128]),
        bbox_coder=dict(
            type='DeltaXYWHTRBBoxCoder', angle_version=angle_version,
            norm_factor=None, edge_swap=True, proj_xy=True,
            target_means=(0.,0.,0.,0.,0.), target_stds=(1.,1.,1.,1.,1.),
            use_box_type=False),
        loss_cls=dict(type='mmdet.FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_bbox=dict(type='mmdet.SmoothL1Loss', beta=0.11, loss_weight=1.0),
    ),

    # ---- refinement stages
    bbox_head_refine=[
        dict(
            type='S2ARefineHead', num_classes=12, in_channels=256,
            stacked_convs=2, feat_channels=256,
            frm_cfg=dict(type='AlignConv', feat_channels=256, kernel_size=3, strides=[8,16,32,64,128]),
            anchor_generator=dict(type='PseudoRotatedAnchorGenerator', strides=[8,16,32,64,128]),
            bbox_coder=dict(
                type='DeltaXYWHTRBBoxCoder', angle_version=angle_version,
                norm_factor=None, edge_swap=True, proj_xy=True,
                target_means=(0.,0.,0.,0.,0.), target_stds=(1.,1.,1.,1.,1.)),
            loss_cls=dict(type='mmdet.FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
            loss_bbox=dict(type='mmdet.SmoothL1Loss', beta=0.11, loss_weight=1.0),
        )
    ],

    train_cfg=dict(
        init=dict(
            assigner=dict(
                type='mmdet.MaxIoUAssigner', pos_iou_thr=0.5, neg_iou_thr=0.4,
                min_pos_iou=0, ignore_iof_thr=-1, iou_calculator=dict(type='RBboxOverlaps2D')),
            allowed_border=-1, pos_weight=-1, debug=False),
        refine=[
            dict(
                assigner=dict(
                    type='mmdet.MaxIoUAssigner', pos_iou_thr=0.5, neg_iou_thr=0.4,
                    min_pos_iou=0, ignore_iof_thr=-1, iou_calculator=dict(type='RBboxOverlaps2D')),
                allowed_border=-1, pos_weight=-1, debug=False),
        ],
        stage_loss_weights=[1.0]),
    test_cfg=dict(
        nms_pre=2000, min_bbox_size=0, score_thr=0.05,
        nms=dict(type='nms_rotated', iou_threshold=0.1), max_per_img=2000)
)

# --- a bit calmer at start; bump later if you like
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=1e-4),
    clip_grad=dict(max_norm=35, norm_type=2),
)

# ---------------- Pipelines with GT filtering (prevents NaNs)
train_pipeline = [
    dict(type='mmdet.LoadImageFromFile'),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.FilterAnnotations', min_gt_bbox_wh=(4, 4), keep_empty=False),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='mmdet.RandomFlip', prob=0.75,
         direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='mmdet.PackDetInputs'),
]

# keep LoadAnnotations for val/test so DOTAMetric has GT; also filter tiny GT
val_pipeline = [
    dict(type='mmdet.LoadImageFromFile'),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.FilterAnnotations', min_gt_bbox_wh=(4, 4), keep_empty=False),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id','img_path','ori_shape','img_shape','scale_factor')),
]

test_pipeline = [
    dict(type='mmdet.LoadImageFromFile'),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.FilterAnnotations', min_gt_bbox_wh=(4, 4), keep_empty=False),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id','img_path','ori_shape','img_shape','scale_factor')),
]

# ---------------- Dataloaders (Windows-safe)
train_dataloader = dict(
    batch_size=2,
    num_workers=0,                 # Windows: safer
    persistent_workers=False,      # must be False when num_workers=0
    dataset=dict(
        type='DOTADataset',
        data_root=data_root,
        ann_file='labelTxt/',
        data_prefix=dict(img_path='images/'),
        metainfo=metainfo,
        pipeline=train_pipeline,
        filter_cfg=dict(filter_empty_gt=True),
    ),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    dataset=dict(
        type='DOTADataset',
        data_root=data_root,
        ann_file='labelTxt/',
        data_prefix=dict(img_path='images/'),
        metainfo=metainfo,
        pipeline=val_pipeline,
        test_mode=True,
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    dataset=dict(
        type='DOTADataset',
        data_root=data_root,
        ann_file='labelTxt/',
        data_prefix=dict(img_path='images/'),
        metainfo=metainfo,
        pipeline=test_pipeline,
        test_mode=True,
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

# Optional: log more frequently
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=10),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'),
)

# evaluation json prefix
test_evaluator = dict(
    type='DOTAMetric',
    metric='mAP',
    outfile_prefix='./work_dirs/dota/s2anet_le90_dota12_Task1'
)

# ==== Paper-matched base training overrides ====
# 12 epochs total; evaluate every epoch
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)

# Batch size 8 (paper)
train_dataloader.update(batch_size=8)

# Optimizer exactly as in the paper
optim_wrapper['optimizer'].update(lr=0.01, momentum=0.9, weight_decay=1e-4)

# LR schedule: 200-iter warm-up then step at 8 & 11
param_scheduler = [
    dict(type='LinearLR', begin=0, end=200, by_epoch=False, start_factor=1/3),
    dict(type='MultiStepLR', begin=0, end=12, by_epoch=True,
         milestones=[8, 11], gamma=0.1),
]