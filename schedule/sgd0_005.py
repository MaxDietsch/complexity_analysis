optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001))

param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[50, 100, 150, 200], gamma=0.5)

train_cfg = dict(by_epoch=True, max_epochs=250, val_interval=1)

val_cfg = dict()
test_cfg = dict()
