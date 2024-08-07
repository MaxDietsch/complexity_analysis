from mmpretrain import ImageClassificationInferencer
import torch
from mmengine.config import Config
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import os
import sys

# for classification of healthy or unhealthy
model = 'icnet128'


model_config = f'../config/{model}.py'
model_pretrained = f'../weights/checkpoint.pth'
image_size = (1024, 1024)


cfg = Config.fromfile(model_config)
model = ImageClassificationInferencer(model = model_config, pretrained = model_pretrained)


# Define the transformations: resize and convert to tensor
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor()
])

file = sys.argv[1]
filename, file_extension = os.path.splitext(file)

image = Image.open(file)
image = image.convert('RGB')
width, height = image.size
tensor = transform(image)
tensor = tensor.unsqueeze(dim = 0)

tup = model.model(tensor)
cly_map = tup[0].squeeze()
label = tup[1].squeeze().item() * 9
print(f'predicted: {tup[1].squeeze() * 9}')

new_path = os.path.join(filename + '-' + str(label) + file_extension)
transform_to_image = transforms.ToPILImage()
image = transform_to_image(cly_map)

with open(filename + '_score.txt', 'w') as score_file:
    score_file.write(str(label))

image = image.convert('L')
image = image.resize((width, height))
image.save(new_path)

        


        
