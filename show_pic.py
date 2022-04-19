import os, glob
import random
import torch
import torch.nn as nn
import  torch.optim as optim
from PIL import Image
from torchvision.transforms import transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from model import *
import math
import matplotlib.pyplot as plt

def get_images_labels(folder):
    dic = {}
    class_id = 0
    for label in os.listdir(folder):
        label_path = folder + '/' + label
        dic[class_id] = os.listdir(label_path)
        class_id += 1
    return dic


# show height
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        # height = 3
        # plt.text(rect.get_x() + rect.get_width() / 2. - 0.2, 1.03 * height, '%s' % int(height))
        plt.text(rect.get_x() + rect.get_width() / 2., 1.03 * height, '%s' % int(height))

if __name__ == '__main__':
    name = 'labels.json'
    import json
    with open(name, 'r') as file_obj:
          dict = json.load(file_obj)

    angle_class = dict['angle_class']
    num_angle_class = len(angle_class)
    tumor_class = dict['tumor_class']
    num_tumor_class = len(tumor_class)
    image_labels = dict['image_labels']

    angle_dic = {}
    for k, v in image_labels.items():
        if v[0] not in angle_dic:
            angle_dic[v[0]] = [k]
        else:
            angle_dic[v[0]].append(k)

    tumor_dic = {}
    for k, v in image_labels.items():
        if v[1] not in tumor_dic:
            tumor_dic[v[1]] = [k]
        else:
            tumor_dic[v[1]].append(k)


    name_list = [key for key in angle_dic.keys()]
    num_list = [len(angle_dic[key]) for key in angle_dic.keys()]
    autolabel(plt.bar(range(len(num_list)), num_list,  tick_label=name_list))
    plt.title("Distribution of different imaging angles in the training set.")
    plt.xlabel("Label")
    plt.ylabel("Number of samples")
    plt.savefig("figures/angles.png")
    plt.show()

    name_list = [key for key in tumor_dic.keys()]
    num_list = [len(tumor_dic[key]) for key in tumor_dic.keys()]
    autolabel(plt.bar(range(len(num_list)), num_list, tick_label=name_list))
    plt.title("Distribution of different tumor types in the training set.")
    plt.xlabel("Label")
    plt.ylabel("Number of samples")
    plt.savefig("figures/types.png")
    plt.show()

    file = 'data/1.jpg'
    im = Image.open(file)
    # resize
    im1 = transforms.Resize((224, 224))(im)
    # horizontal flip
    im3 = transforms.RandomHorizontalFlip(p=0.8)(im1)
    plt.subplot(131)
    plt.title("srcImage")
    plt.imshow(im)
    plt.subplot(132)
    plt.title("Resize")
    plt.imshow(im1)
    plt.subplot(133)
    plt.title("RandomHorizontal")
    plt.imshow(im3)
    plt.show()


