
### do inference with a single pictures as cli argument
### output is stored in the same directory as the input directory is
### output is 1 complexity map and 1 txt file where the complexity score is stored 

## the loaded checkpoint weights should be in a file ../../weights/model_checkpoint.pth relative to this file



from icnet import ICNet
import numpy as np
import sys
import torchvision.transforms as transforms
import torch
import os 
from PIL import Image


IMAGE_SIZE = (1024, 1024)

#model = ICNet()

model = torch.load('../../weights/model_checkpoint.pth')
model.eval()


transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor()
])

file = sys.argv[1]
filename, file_extension = os.path.splitext(file)

image = Image.open(file)
image = image.convert('RGB')
width, height = image.size
tensor = transform(image)
tensor = tensor.unsqueeze(dim = 0)

tup = model(tensor)
cly_map = tup[0].squeeze()
label = tup[1].squeeze().item() * 9
print(f'predicted: {tup[1].squeeze() * 9}')

new_path = os.path.join(filename + '-' + 'act_map' + file_extension)
transform_to_image = transforms.ToPILImage()
image = transform_to_image(cly_map)

with open(filename + '_score.txt', 'w') as score_file:
    score_file.write(str(label))

image = image.convert('L')
image = image.resize((width, height))
image.save(new_path)
