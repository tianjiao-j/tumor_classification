import os, glob
import random
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision.transforms import transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from model import *
import math
import json


def get_image_dic(filename):
    with open(filename, 'r') as file_obj:
        dict = json.load(file_obj)
    return dict


def split_train_val(image_dic, val_ratio=0.2, seed=0):
    if seed is not None:
        random.seed(seed)

    angle_class = image_dic['angle_class']
    num_angle_class = len(angle_class)
    tumor_class = image_dic['tumor_class']
    num_tumor_class = len(tumor_class)
    image_labels = image_dic['image_labels']

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

    train_images, train_labels = [], []
    val_images, val_labels = [], []
    for a_k, a_v in angle_dic.items():
        for t_k, t_v in tumor_dic.items():
            a_label = [1 if angle_class[i] == a_k else 0 for i in range(num_angle_class)]
            t_label = [1 if tumor_class[i] == t_k else 0 for i in range(num_tumor_class)]
            # new_label = a_label + t_label
            new_label = t_label + a_label
            intersection = list(set(a_v) & set(t_v))
            val_image = random.sample(intersection, int(len(intersection) * val_ratio))
            val_label = [new_label for _ in range(len(val_image))]

            train_image = list(set(intersection) - set(val_image))
            train_label = [new_label for _ in range(len(train_image))]

            train_images += train_image
            train_labels += train_label
            val_images += val_image
            val_labels += val_label
    return train_images, train_labels, val_images, val_labels


class MyDataSet(Dataset):
    """define dataset"""

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        # reference:
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        # print("batch size : ", batch.size())
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels


if __name__ == '__main__':
    import numpy as np

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    file = 'labels.json'
    dic = get_image_dic(file)
    train_images, train_labels, val_images, val_labels = split_train_val(dic, val_ratio=0.2)
    print(len(train_labels[0]))
    print(np.array(train_labels).shape)
    print(np.array(val_labels).shape)


    # get path for images
    root = 'data/'
    train_images_path = [root + img for img in train_images]
    val_images_path = [root + img for img in val_images]

    # pre-processing
    data_transform = {
        "train": transforms.Compose([
            # resize
            transforms.Resize((256, 256)),
            # data argumentation
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # normalization
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize((256, 256)),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    train_data_set = MyDataSet(images_path=train_images_path,
                               images_class=train_labels,
                               transform=data_transform["train"])

    val_data_set = MyDataSet(images_path=val_images_path,
                             images_class=val_labels,
                             transform=data_transform["val"])
    print(len(train_data_set))

    batch_size = 32
    sum_train_item = len(train_images_path) * 2
    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               collate_fn=train_data_set.collate_fn)

    sum_val_item = len(val_images_path) * 2
    val_loader = torch.utils.data.DataLoader(val_data_set,
                                             batch_size=batch_size,
                                             collate_fn=val_data_set.collate_fn)

    net = resnet50()
    inchannel = net.fc.in_features
    net.fc = nn.Linear(inchannel, 7)
    net_dict = net.state_dict()
    model_weight_path = "resnet50.pth"

    # load weights
    pretrained_dict = torch.load(model_weight_path)
    new_dict = {k: v for k, v in pretrained_dict.items() if not k.startswith("fc")}
    # update parameters
    net_dict.update(new_dict)
    net.load_state_dict(net_dict)
    net = net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=0.0001)  # learning rate
    criterion = nn.BCELoss()
    epochs = 5000

    Sigmoid_fun = nn.Sigmoid()
    f = open('resnet50/log.txt', 'w')
    best_val_acc = 0.

    for i in range(epochs):
        cost = 0
        correct = 0.
        for step, data in enumerate(train_loader):
            images, labels = data
            images, labels = images.to(device), labels.to(device).float()
            _, true = labels.topk(2, 1, True, True)
            true = torch.sort(true, dim=1)[0]
            outputs = net(images)
            s_outputs = Sigmoid_fun(outputs)
            # calculate train acc
            pred1 = torch.argmax(s_outputs[:, :4], dim=1)
            pred2 = torch.argmax(s_outputs[:, 4:], dim=1) + 4
            pred = torch.stack([pred1, pred2], dim=1)
            correct += torch.sum(true == pred)

            # calculate train loss
            loss = criterion(s_outputs, labels)
            cost += loss
            loss.backward()  # loss backpropagation
            optimizer.step()

        train_cost = cost / (step + 1)
        train_acc = correct / sum_train_item

        with torch.no_grad():
            val_correct = 0.
            val_cost = 0.
            for step, data in enumerate(val_loader):
                images, labels = data
                images, labels = images.to(device), labels.to(device).float()
                outputs = net(images)
                _, true = labels.topk(2, 1, True, True)
                true = torch.sort(true, dim=1)[0]
                s_outputs = Sigmoid_fun(outputs)
                # calculate val acc
                pred1 = torch.argmax(s_outputs[:, :4], dim=1)
                pred2 = torch.argmax(s_outputs[:, 4:], dim=1) + 4
                pred = torch.stack([pred1, pred2], dim=1)

                val_correct += torch.sum(true == pred)
                # calculate val loss
                loss = criterion(s_outputs, labels)
                val_cost += loss
            val_acc = val_correct / sum_val_item
            val_loss = val_cost / (step + 1)

        state = {'net': net.state_dict(), 'epoch': i}
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            torch.save(state, 'resnet50/best.pt')

        if (i + 1) % 500 == 0:
            torch.save(state, 'resnet50/epoch_{}.pt'.format(i))

        s = 'epoch:{} train loss:{:.7f} train acc:{:.7f} val loss:{:.7f} val acc:{:.7f}'.format(
            i, train_cost, train_acc, val_loss, val_acc
        )
        print(s)
        f.write(s + '\n')
        f.flush()
    torch.save(state, 'last.pt')
