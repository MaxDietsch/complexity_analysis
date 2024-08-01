
### load the mmpretrain model file and save it again (but just the state_dict)
### this makes it possible for pytorch to load the checkpoint file

from icnet import ICNet
import torch
from icnet import ICNet

IMAGE_SIZE = (1024, 1024)

model = ICNet()

checkpoint = torch.load('../../weights/checkpoint.pth')

model.load_state_dict(checkpoint['state_dict'])

torch.save(model, '../../weights/model_checkpoint.pth')
