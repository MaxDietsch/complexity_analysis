from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmpretrain.evaluation.metrics import Accuracy
from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample


# spatial dim must be divider of height and width of input
class slam(nn.Module):
    def __init__(self, spatial_dim):
        super(slam,self).__init__()
        self.spatial_dim = spatial_dim
        self.linear = nn.Sequential(
             nn.Linear(spatial_dim**2,512),
             nn.ReLU(),
             nn.Linear(512,1),
             nn.Sigmoid()
        )

    def forward(self, feature):
        n,c,h,w = feature.shape
        if (h != self.spatial_dim):
            x = F.interpolate(feature,size=(self.spatial_dim,self.spatial_dim),mode= "bilinear", align_corners=True)
        else:
            x = feature


        x = x.view(n,c,-1) # (b, c, spatial_dim**2)
        x = self.linear(x) # (b, c, 1)
        x = x.unsqueeze(dim =3) # (b, c, 1, 1) -> (expand_as) -> (b, c, h, w)
        out = x.expand_as(feature)*feature 

        return out


class to_map(nn.Module):
    def __init__(self,channels):
        super(to_map,self).__init__()
        self.to_map = nn.Sequential(
            nn.Conv2d(in_channels=channels,out_channels=1, kernel_size=1,stride=1),
            nn.Sigmoid()
        )
        
    def forward(self,feature):
        return self.to_map(feature)
    
        
class conv_bn_relu(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size = 3, padding = 1, stride = 1):
        super(conv_bn_relu,self).__init__()
        self.conv = nn.Conv2d(in_channels= in_channels, out_channels= out_channels, kernel_size= kernel_size, padding= padding, stride = stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


@MODELS.register_module()
class ICNetHead128(BaseModule):


    def __init__(self,
             loss: dict = dict(type='ICNetLoss'),
             topk: Union[int, Tuple[int]] = (1, ),
             cal_acc: bool = False,
             init_cfg: Optional[dict] = None,
             size_slam = 128,
             num_scores = 10):
        super(ICNetHead128, self).__init__(init_cfg=init_cfg)

        self.topk = topk
        if not isinstance(loss, nn.Module):
            loss = MODELS.build(loss)
        self.loss_module = loss
        self.cal_acc = cal_acc
        self.size_slam = size_slam
        self.num_scores = num_scores

        # map prediction
        self.to_map_f = conv_bn_relu(256*2,256*2)
        self.to_map_f_slam = slam(self.size_slam)
        self.to_map = to_map(256*2)

        ## score prediction head
        self.to_score_f = conv_bn_relu(256*2,256*2)
        self.to_score_f_slam = slam(self.size_slam)
        self.head = nn.Sequential(
            nn.Linear(256*2,512),
            nn.ReLU(),
            nn.Linear(512, 1), nn.Sigmoid()
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
    
    def pre_logits(self, feats: Tuple[torch.Tensor]):
        print('pre_logits was called which is not implemented for regression head of ICNet')

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """The forward process."""
        cly_map = self.to_map_f(feats) #(b, 512, 128, 128)
        #print(cly_map.shape)

        cly_map = self.to_map_f_slam(cly_map) #(b, 512, 128, 128)
        #print(cly_map.shape)

        cly_map = self.to_map(cly_map) #(b, 1, 128, 128)
        #print(cly_map.shape)

        detail_score = self.to_score_f(feats) #(b, 512, 128, 128)
        #print(score_feature.shape)

        detail_score = self.to_score_f_slam(detail_score) #(b, 512, 256, 256)
        #print(score_feature.shape)

        detail_score = self.avgpool(detail_score) #(b, 512, 1, 1)
        #print(score_feature.shape)

        detail_score = detail_score.squeeze() #(b, 512)
        #print(score_feature.shape)

        detail_score = self.head(detail_score) #(b, 1)
        #print(score.shape)

        outs = [cly_map, detail_score]

        return outs

    def loss(self, feats: torch.Tensor, data_samples: List[DataSample],
             **kwargs) -> dict:
        # The part can be traced by torch.fx
        cls_score = self(feats)

        # The part can not be traced by torch.fx
        losses = self._get_loss(cls_score, data_samples)
        return losses

    def _get_loss(self, cls_score: torch.Tensor,
                  data_samples: List[DataSample]):
        """Unpack data samples and compute loss."""
        # Unpack data samples and pack targets
        if 'gt_score' in data_samples[0]:
            # Batch augmentation may convert labels to one-hot format scores.
            target = torch.stack([i.gt_score for i in data_samples]).to(torch.float)
        else:
            target = torch.cat([i.gt_label for i in data_samples]).to(torch.float)
            
        target /= (self.num_scores - 1)
        # compute loss
        losses = dict()
        loss = self.loss_module(
            cls_score, target)
        losses['loss'] = loss

        # compute accuracy
        if self.cal_acc:
            print('accuracy is calculated in _get_loss of ICNet Head, maybe need to adjust it')
            assert target.ndim == 1, 'If you enable batch augmentation ' \
                'like mixup during training, `cal_acc` is pointless.'
            acc = Accuracy.calculate(cls_score, target, topk=self.topk)
            losses.update(
                {f'accuracy_top-{k}': a
                 for k, a in zip(self.topk, acc)})

        return losses

    def predict(
        self,
        feats: Tuple[torch.Tensor],
        data_samples: Optional[List[Optional[DataSample]]] = None
    ) -> List[DataSample]:
        # The part can be traced by torch.fx
        cls_score = self(feats)
        cly_map = cls_score[0] 
        cls_score = cls_score[1] * (self.num_scores - 1) 

        # The part can not be traced by torch.fx
        predictions = self._get_predictions(cls_score, data_samples)
        return predictions, cly_map

    def _get_predictions(self, cls_scores, data_samples):

        out_data_samples = []
        if data_samples is None:
            data_samples = [None for _ in range(cls_scores.size(0))]

        for data_sample, score in zip(data_samples, cls_scores):
            if data_sample is None:
                data_sample = DataSample()

            data_sample.set_pred_score(score)
            out_data_samples.append(data_sample)
        return out_data_samples
