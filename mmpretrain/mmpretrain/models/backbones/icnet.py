import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights

from mmpretrain.registry import MODELS
from .base_backbone import BaseBackbone


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

        

class up_conv_bn_relu(nn.Module):
    def __init__(self,up_size, in_channels, out_channels = 64, kernal_size = 1, padding =0, stride = 1):
        super(up_conv_bn_relu,self).__init__()
        self.upSample = nn.Upsample(size = (up_size,up_size),mode="bilinear",align_corners=True)
        self.conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size = kernal_size, stride = stride, padding= padding)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.act = nn.ReLU()
        
    def forward(self,x):
        x = self.upSample(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


@MODELS.register_module()
class ICNetBackboneRes18Out128(BaseBackbone):
    def __init__(self, 
                 image_size = 1024, 
                 size_slam = 128, 
                 init_cfg = None,
                 norm_eval = False, 
                 out_dim = 256):

        super(ICNetBackboneRes18Out128, self).__init__(init_cfg)
        resnet18Pretrained1 = torchvision.models.resnet18(weights = ResNet18_Weights.IMAGENET1K_V1)
        resnet18Pretrained2 = torchvision.models.resnet18(weights = ResNet18_Weights.IMAGENET1K_V1)
        
        self.image_size = image_size
        self.size_slam = size_slam
        self.norm_eval = norm_eval
        self.out_dim = out_dim
        
        ## detail branch
        self.b1_1 = nn.Sequential(*list(resnet18Pretrained1.children())[:5])  
        self.b1_1_slam = slam(self.size_slam)
    
        self.b1_2 = list(resnet18Pretrained1.children())[5]
        self.b1_2_slam = slam(self.size_slam)

        ## context branch
        self.b2_1 = nn.Sequential(*list(resnet18Pretrained2.children())[:5])
        self.b2_1_slam = slam(self.size_slam)
        
        self.b2_2 = list(resnet18Pretrained2.children())[5]
        self.b2_2_slam = slam(self.size_slam)
    
        self.b2_3 = list(resnet18Pretrained2.children())[6]
        self.b2_3_slam = slam(self.size_slam // 2)
        
        self.b2_4 = list(resnet18Pretrained2.children())[7]
        self.b2_4_slam = slam(self.size_slam // 4)

        ## upsample
        self.upsize = image_size // 8
        self.up1 = up_conv_bn_relu(up_size = self.upsize, in_channels = 128, out_channels = 256)
        self.up2 = up_conv_bn_relu(up_size = self.upsize, in_channels = 512, out_channels = 256)

        # map prediction
        self.to_map_f = conv_bn_relu(256*2,256*2)
        self.to_map_f_slam = slam(self.size_slam)
        self.to_map = to_map(256*2)
        #map scoring
        self.map_to_vec = nn.Linear(128 * 128, self.out_dim // 2)

        ## score prediction head
        self.to_score_f = conv_bn_relu(256*2,256*2)
        self.to_score_f_slam = slam(self.size_slam)
        self.head = nn.Sequential(
            nn.Linear(256*2,512),
            nn.ReLU(),
            nn.Linear(512, self.out_dim // 2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
       
    
    def forward(self, x1):
        assert(x1.shape[2] == x1.shape[3] == self.image_size)
        x2 = F.interpolate(x1, size= (self.image_size // 2,self.image_size // 2), mode = "bilinear", align_corners= True)
        
        # detail branch
        x1 = self.b1_1(x1) #(b, 64, 256, 256)
        
        #print(x1.shape)
        x1 = self.b1_1_slam(x1) #(b, 64, 256, 256)
        #print(x1.shape)

        x1 = self.b1_2(x1) #(b, 128, 128, 128)
        #print(x1.shape)
     
        x1 = self.b1_2_slam(x1) #(b, 128, 128, 128)
        
        # context branch
        x2 = self.b2_1(x2) #(b, 64, 128, 128)
        #print(x2.shape)
        
        x2 = self.b2_1_slam(x2) #(b, 64, 128, 128)
        #print(x2.shape)

        x2 = self.b2_2(x2) #(b, 128, 64, 64)
        #print(x2.shape)
       
        x2 = self.b2_2_slam(x2) #(b, 128, 64, 64)
        #print(x2.shape)

        x2 = self.b2_3(x2) #(b, 256, 32, 32)
        #print(x2.shape)

        x2 = self.b2_3_slam(x2) #(b, 256, 32, 32)
        #print(x2.shape)
        
        x2 = self.b2_4(x2) #(b, 512, 16, 16)
        #print(x2.shape)
        
        x2 = self.b2_4_slam(x2) #(b, 512, 16, 16)
        #print(x2.shape)

        x1 = self.up1(x1) #(b, 256, 128, 128)
        #print(x1.shape)

        x2 = self.up2(x2) #(b, 256, 128, 128)
        #print(x2.shape)

        x_cat = torch.cat((x1, x2), dim = 1) #(b, 512, 128, 128)
        #print(x_cat.shape)

        cly_map = self.to_map_f(x_cat) #(b, 512, 128, 128)
        #print(cly_map.shape)

        cly_map = self.to_map_f_slam(cly_map) #(b, 512, 128, 128)
        #print(cly_map.shape)

        cly_map = self.to_map(cly_map) #(b, 1, 128, 128)
        #print(cly_map.shape)

        map_score = cly_map.view(cly_map.shape[0], -1) #(b, 128 * 128)
        #print(map_score)

        map_score = self.map_to_vec(map_score) #(b, self.out_dim // 2)
        #print(map_score)

        detail_score = self.to_score_f(x_cat) #(b, 512, 128, 128)
        #print(score_feature.shape)

        detail_score = self.to_score_f_slam(detail_score) #(b, 512, 256, 256)
        #print(score_feature.shape)

        detail_score = self.avgpool(detail_score) #(b, 512, 1, 1)
        #print(score_feature.shape)

        detail_score = detail_score.squeeze() #(b, 512)
        #print(score_feature.shape)

        detail_score = self.head(detail_score) #(b, self.out_dim // 2)
        #print(score.shape)

        score = torch.cat((detail_score, map_score), dim = 1)
        #print(score.shape)

        out = [cly_map, score]

        return tuple(outs)


    def train(self, mode = True):
        super(ICNetBackboneRes18Out128, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

    def init_weights(self):
        super(ICNetBackboneRes18Out128, self).init_weights()

