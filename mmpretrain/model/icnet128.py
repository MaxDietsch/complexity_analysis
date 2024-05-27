model = dict(
    type='ImageClassifier',
    backbone=dict(type='ICNetBackboneRes18Out128', image_size = 1024, size_slam = 128),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=4,
        in_channels=1792,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1),
    ))
