model = dict(
    type='ICNetRegressor',
    backbone=dict(type='ICNetBackboneRes18Out128', image_size = 1024, size_slam = 128, norm_eval = False),
    neck=None,
    head=dict(
        type='ICNetHead128',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 3),
    ))




