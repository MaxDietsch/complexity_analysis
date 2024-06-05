_base_ = [
        '../model/icnet128_pretrain.py',
        '../data/pretrain8.py',
        '../schedule/sgd0_005.py',
        '../runtime/default.py'
        ]

load_from = None
resume = False
