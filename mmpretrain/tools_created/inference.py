from mmpretrain import ImageClassificationInferencer
import torch
from mmengine.config import Config
import numpy as np

# for classification of healthy or unhealthy
model = 'icnet128'
epoch = '12'


model_config = f'../tools/work_dirs/{model}/{model}.py'
model_pretrained = f'../tools/work_dirs/{model}/epoch_{epoch}.pth'


cfg = Config.fromfile(model_config)
model = ImageClassificationInferencer(model = model_config, pretrained = model_pretrained)

paths, labels = [], []

with open("../../../dataset_default/meta/test.txt", "r") as file:
    for line in file:
        path, label = line.strip().split(" ", 1)
        label = int(label)
        tup = model.backbone(path)
        cly_map = tup[0]
        print(path)


            


