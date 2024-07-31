from icnet import ICNet
import numpy as np
import sys
import torchvision.transforms as transforms

IMAGE_SIZE = (1024, 1024)

model = ICNet()

model.load_state_dict(torch.load('../weights/checkpoint.pth'))


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
