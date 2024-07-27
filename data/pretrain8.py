#construct dataloader and evaluator
dataset_type = 'CustomDataset'
data_preprocessor = dict(
            # Input image data channels in 'RGB' order
                mean=[135.39, 109.55, 102.62],
                    std=[76.27, 74.48, 77.86],
                        to_rgb=True,
        )

train_pipeline = [
            dict(type='LoadImageFromFile'),     # read image
            dict(type = 'RandomFlip', prob = 0.5, direction=['horizontal', 'vertical']),
            dict(type = 'RandomApply', transforms = [dict(type = 'ColorJitter', brightness = [0.8, 1.3], contrast = [0.8, 1.3], saturation = [0.8, 1.3], hue = 0.1, backend = 'pillow')], prob = 0.8),
            dict(type='Resize', scale=(1024, 1024), interpolation='bicubic'),
            dict(type='PackInputs'),         # prepare images and labels
        ]

test_pipeline = [
            dict(type='LoadImageFromFile'),     # read image
            dict(type='Resize', scale=(1024, 1024), interpolation='bicubic'),
            dict(type='PackInputs'),                 # prepare images and labels
        ]

train_dataloader = dict(
            batch_size=8,
            num_workers=8,
            dataset=dict(
                type=dataset_type,
                data_root='../../../dataset_pretrain',
                ann_file='meta/train.txt',
                data_prefix='images',
                with_label=True,
                classes=['0', '1', '2', '3', '4'],
                pipeline=train_pipeline),
            sampler=dict(type='DefaultSampler', shuffle=True),
            persistent_workers=True,
        )

val_dataloader = dict(
            batch_size=8,
            num_workers=8,
            dataset=dict(
                type=dataset_type,
                data_root='../../../dataset_pretrain',
                ann_file='meta/test.txt',
                data_prefix='images',
                with_label=True,
                classes=['0', '1', '2', '3', '4'],
                pipeline=test_pipeline),
            sampler=dict(type='DefaultSampler', shuffle=False),
            persistent_workers=True,
        )

val_evaluator = [
        dict(type='ICNetMAE'),
        ]




test_dataloader = val_dataloader
test_evaluator = val_evaluator

