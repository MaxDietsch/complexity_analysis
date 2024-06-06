model = dict(
    type='ICNetRegressor',
    backbone=dict(type='ICNetBackboneRes18Out128', image_size = 1024, size_slam = 128, norm_eval = False, init_cfg = dict(type = 'Pretrained', checkpoint='../work-dir/pretrain_w0.5/epoch_250.pth', prefix = 'backbone')),
    neck=None,
    head=dict(
        type='ICNetHead128',
        loss=dict(type='ICNetLoss', num_scores = 10, map_weighting = 0.5),
        num_scores = 10,
    ))




