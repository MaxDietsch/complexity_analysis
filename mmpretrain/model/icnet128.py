model = dict(
    type='ImageClassifier',
    backbone=dict(type='ICNetBackboneRes18Out128', image_size = 1024, size_slam = 128, norm_eval = False, out_dim = 256),
    neck=None,
    head=dict(
        type='LinearClsHead',
        num_classes=10,
        in_channels=256,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 3),
    ))
