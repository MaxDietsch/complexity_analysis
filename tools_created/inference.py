from mmpretrain import ImageClassificationInferencer
import torch
from mmengine.config import Config
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import os


# for classification of healthy or unhealthy
model = 'icnet128'
epoch = '250'
weight = 0.9


model_config = f'../config/{model}.py'
model_pretrained = f'../work-dir/w_{weight}/epoch_{epoch}.pth'
output_dir = '../../plain_torch/output_images'
image_size = (1024, 1024)


cfg = Config.fromfile(model_config)
model = ImageClassificationInferencer(model = model_config, pretrained = model_pretrained)


# Define the transformations: resize and convert to tensor
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor()
])



with open("../../../dataset_default/meta/test.txt", "r") as file:
    i = 0
    for line in file:
        path, label = line.strip().split(" ", 1)
        filename = os.path.basename(path)
        filename, file_extension = os.path.splitext(filename)

        image = Image.open('../' + path)
        image = image.convert('RGB')
        width, height = image.size
        tensor = transform(image)
        tensor = tensor.unsqueeze(dim = 0)
        print(tensor.shape)

        label = int(label)
        tup = model.model(tensor)
        cly_map = tup[0].squeeze()
        print(f'truth: {label} \t predicted: {tup[1].squeeze() * 9}')

        new_path = os.path.join(output_dir, filename + '-' + str(label) + file_extension)
        print(new_path)
        transform_to_image = transforms.ToPILImage()
        image = transform_to_image(cly_map)

        image = image.convert('L')
        image = image.resize((width, height))
        image.save(new_path)

        #i += 1 
        #if i == 171:
        #    break



        
