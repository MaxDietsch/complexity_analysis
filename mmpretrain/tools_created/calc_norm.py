import cv2
import os

"""
This script takes a directory of images and calculates the
mean of each channel and the standard deviation of each channel of the overall image dataset
This is useful for noramlization.
"""

def calc_norm(dir):
    sum_r, sum_g, sum_b = 0.0, 0.0, 0.0
    sum_sq_diff_r, sum_sq_diff_g, sum_sq_diff_b = 0.0, 0.0, 0.0
    i = 0
    num_images = 0
    #when normalize with data of val and test
    #for dir in dirs:
    #   for img_name in os.listdir(dir)
    for img_name in os.listdir(dir):
        print(i)
        i += 1
        num_images += 1
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            continue

        img_path = os.path.join(dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        sum_r += image[:, :, 0].mean()
        sum_g += image[:, :, 1].mean()
        sum_b += image[:, :, 2].mean()


    mean_r = sum_r / num_images
    mean_g = sum_g / num_images
    mean_b = sum_b / num_images

    #for dir in dirs:
    #    for img_name in os.listdir(dir):
    for img_name in os.listdir(dir):
        print(i)
        i += 1
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            continue

        img_path = os.path.join(dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        sum_sq_diff_r += ((image[:, :, 0] - mean_r) ** 2).mean()
        sum_sq_diff_g += ((image[:, :, 1] - mean_g) ** 2).mean()
        sum_sq_diff_b += ((image[:, :, 2] - mean_b) ** 2).mean()


    var_r = sum_sq_diff_r / num_images
    var_g = sum_sq_diff_g / num_images
    var_b = sum_sq_diff_b / num_images

    std_r = var_r ** 0.5
    std_g = var_g ** 0.5
    std_b = var_b ** 0.5

    print(f"mean values: r: {mean_r}, g: {mean_g}, b: {mean_b}")
    print(f"standard deviations: r: {std_r}, g: {std_g}, b: {std_b}")


data_dir = '../../../dataset_pretrain/images'
#directories =[os.path.join(data_dir, dir) for dir in os.listdir(data_dir)]
#print(directories)
calc_norm(data_dir)
