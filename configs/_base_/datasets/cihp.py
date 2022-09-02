# dataset settings
dataset_type = 'CIHP'
data_root = 'data/CIHP/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
flip_map = ([14, 15], [16, 17], [18, 19])
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_parsing=True, with_seg=True),
    dict(type='Resize',
         img_scale=[(1400, 512), (1400, 640), (1400, 704),
                    (1400, 768), (1400, 800), (1400, 864)],
         multiscale_mode='value',
         keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5, flip_map=flip_map),
    dict(type='LoadPartAnno', num_parse_classes=20),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img',
                               'gt_bboxes',
                               'gt_labels',
                               'gt_parsings',
                               'gt_parsing_labels',
                               'gt_part_points',
                               'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/CIHP_train.json',
        img_prefix=data_root + 'train_img/',
        seg_prefix=data_root + 'train_seg/',
        parsing_prefix=data_root + 'train_parsing/',
        pipeline=train_pipeline,
        ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/CIHP_val.json',
        img_prefix=data_root + 'val_img/',
        seg_prefix=data_root + 'val_seg/',
        parsing_prefix=data_root + 'val_parsing/',
        pipeline=test_pipeline,
        ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/CIHP_val.json',
        img_prefix=data_root + 'val_img/',
        seg_prefix=data_root + 'val_seg/',
        parsing_prefix=data_root + 'val_parsing/',
        pipeline=test_pipeline,
    ))