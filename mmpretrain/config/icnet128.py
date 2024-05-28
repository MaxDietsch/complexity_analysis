_base_ = [
        '../model/icnet128.py',
        '../data/default8.py',
        '../schedule/sgd0_005.py',
        '../runtime/default.py'
        ]

load_from = None
resume = False
