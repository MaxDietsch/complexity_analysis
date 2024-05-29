import torch.nn as nn
import torch.nn.functional as F

from mmpretrain.registry import MODELS


@MODELS.register_module()
class ICNetLoss(nn.Module):

    def __init(self, 
               map_weighting = 0.1):
        super(ICNetLoss, self).__init__()

        self.map_weighting = map_weighting

    def forward(self, 
                cls_score,
                label):
        print(cls_score)
        print(label)
