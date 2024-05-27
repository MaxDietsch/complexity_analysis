import os
import torch
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from options import args
from ICNet_bigger_bigger import ICNet

# Load the pre-trained model
model = ICNet(args.image_size, 256)
state_dict = torch.load('checkpoint/ck_bigger_bigger100_bs8_is1024_lr0.005.pth', map_location = 'cuda:0')

new_state_dict = {}
for key, value in state_dict.items():
    new_key = key.replace('module.', '')  # Remove 'module.' prefix if present
    new_state_dict[new_key] = value

model.load_state_dict(new_state_dict)
#model.load_state_dict(torch.load('checkpoint/ck_bigger_bigger100_bs8_is1024_lr0.005.pth', map_location = 'cuda:0'))
device = torch.device('cuda')
model.to(device)
#model.eval()

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size)),  # Resize the input image
    transforms.ToTensor(),  # Convert the image to a tensor
])

tensor_transform = transforms.ToTensor()

# Function to read an image, perform inference, and save the output
def process_image(image_path, output_path, upscale_shape = (args.image_size, args.image_size)):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
   
   # Perform inference
    with torch.no_grad():
        _, cly_map = model(input_tensor)

    output_image = transforms.ToPILImage()(cly_map.squeeze())
    output_image.save(output_path)

    # Upscale the output image using OpenCV
    output_image_cv = cv2.imread(output_path)
    height, width = output_image_cv.shape[:2]
    upscaled_image = cv2.resize(output_image_cv, tensor_transform(image).shape[ : 0: -1], interpolation=cv2.INTER_CUBIC)
    #upscaled_image = cv2.resize(output_image_cv, upscale_shape, interpolation=cv2.INTER_CUBIC)


    # Save the upscaled image
    upscaled_output_path = os.path.splitext(output_path)[0] + '_upscaled.jpg'
    cv2.imwrite(upscaled_output_path, upscaled_image)

    print(f'Processed image to {image_path}')

# Main function to process a directory of images
def main(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            process_image(input_path, output_path, )



if __name__ == '__main__':
    input_directory = 'test_images'
    output_directory = 'output_images'
    main(input_directory, output_directory)

