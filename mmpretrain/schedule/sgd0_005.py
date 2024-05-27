optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001))

train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)

val_cfg = dict()
test_cfg = dict()
