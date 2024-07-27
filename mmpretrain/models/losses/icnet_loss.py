import torch.nn as nn
import torch.nn.functional as F

from mmpretrain.registry import MODELS


@MODELS.register_module()
class ICNetLoss(nn.Module):

    def __init__(self, num_scores, map_weighting = 0.1):
        super(ICNetLoss, self).__init__()

        self.num_scores = num_scores
        self.map_weighting = map_weighting
        self.loss_function = nn.MSELoss()

    def forward(self, 
                cls_score,
                label):
        score1 = cls_score[1].squeeze()
        score2 = cls_score[0].mean(axis = (1, 2, 3))

        loss1 = self.loss_function(score1 * (self.num_scores - 1), label * (self.num_scores - 1))
        loss2 = self.loss_function(score2 * (self.num_scores - 1), label * (self.num_scores - 1))
        loss = (1 - self.map_weighting) * loss1 + self.map_weighting * loss2
        return loss
         
