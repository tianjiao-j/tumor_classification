import torch
from PIL import Image
import numpy as np
import json
import os, glob
from model import *
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
from metric import Metric

test_dir = 'data/testing_set'
name = 'labels.json'

with open(name, 'r') as file_obj:
    dict = json.load(file_obj)

angle_class = dict['angle_class']
num_angle_class = len(angle_class)
tumor_class = dict['tumor_class']
num_tumor_class = len(tumor_class)

weight_path = 'resnet34/best.pt'
net = resnet34()
# weight_path = 'resnet50/best.pt'
# net = resnet50()
inchannel = net.fc.in_features
net.fc = nn.Linear(inchannel, 7)
net.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu'))['net'])

tf = transforms.Compose([transforms.Resize((256, 256)),
                         transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
Sigmoid_fun = nn.Sigmoid()
correct = 0.
sum_item = 0.
X, X_labels = [], []
Y, Z = [], []
for t_c in os.listdir(test_dir):
    for a_c in os.listdir(test_dir + '/' + t_c):
        for jpg_path in glob.glob(os.path.join(test_dir, t_c, a_c) + '/*.jpg'):
            # imaging angle - 3 classes; tumor type - 4 classes
            label = [angle_class.index(a_c), tumor_class.index(t_c) + 3]
            im = Image.open(jpg_path)
            x = tf(im).unsqueeze(0)
            s_outputs = net(x)
            # calculate acc
            pred1 = torch.argmax(s_outputs[:, :3], dim=1)
            pred2 = torch.argmax(s_outputs[:, 3:], dim=1)
            pred = [pred1[0].item(), pred2[0].item() + 3]
            correct += sum([label[i] == pred[i] for i in range(2)])
            sum_item += 2
            print("correctly predicted: {} number of labels: {}  accuracy: {}".format(correct, sum_item,
                                                                                      correct / sum_item))
            Y.append(label)
            Z.append(pred)

# output
label_set = [0, 1, 2, 3, 4, 5, 6]
M = Metric(Y, Z, label_set)
print("precision : ", M.Precision())
print("recall :", M.Recall())
print("f1 score : ", M.F1())
print("EMR", M.EMR())
print("micro-F1 : ", M.microF1())
print("micro-Recall : ", M.microRecall())
print("micro-Precision: ", M.microPrecision())
print("HL : ", M.HL())
print("macro-Precision: ", M.macroPrecision())
print("macro-F1: ", M.macroF1())
print("macro-Recall :", M.macroRecall())

