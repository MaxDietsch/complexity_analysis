_base_ = [
        '../model/icnet128.py',
        '../data/default.py',
        '../schedule/sgd0_005.py',
        '../runtime/default.py'
        ]

load_from = None
resume = False
