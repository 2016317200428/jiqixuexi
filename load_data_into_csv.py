# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 16:32:12 2020

@author: jm
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import cv2
import csv

R_B_data = []
image_data_label = []
parent_dir = 'F:/suger_new/images_data/'
IMAGE_SIZE = 1024
R_data = []
G_data = []
B_data = []

def resize_image(image, height=IMAGE_SIZE, width=IMAGE_SIZE):
    top, bottom, left, right = (0, 0, 0, 0)

    h, w, _ = image.shape

    longest_edge = max(h, w)
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass

    BLACK = [0, 0, 0]

    constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)

    return cv2.resize(constant, (height, width))


def compute(path):
    img = cv2.imread(path)
    img = resize_image(img)
    #img = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV) 加载HSV数据
    npim = np.zeros((IMAGE_SIZE,IMAGE_SIZE), dtype=np.float)
    sum_rgb = 0
    npim[:] = img[:, :, 0] + img[:, :, 1] + img[:, :, 2]

    for i in range(1024):
        for j in range(1024):
            if npim[i, j] != 0:
                sum_rgb = sum_rgb + 1

    B_mean = np.sum(img[:, :, 0])/sum_rgb
    G_mean = np.sum(img[:, :, 1])/sum_rgb
    R_mean = np.sum(img[:, :, 2])/sum_rgb

    return R_mean, G_mean, B_mean


def read_path(R, G, B,  labels, path_name):
    for dir_item in os.listdir(path_name):
        label = float(dir_item.split(' ')[0])
        full_path = os.path.abspath(os.path.join(path_name, dir_item))
        r, g, b = compute(full_path)

        R.append(r)
        G.append(g)
        B.append(b)
        labels.append(label)


def load_data(parent_dir):
    R = []
    G = []
    B = []
    labels = []

    read_path(R, G, B, labels, parent_dir + "13")
    read_path(R, G, B, labels, parent_dir + "14")
    read_path(R, G, B, labels, parent_dir + "15")
    read_path(R, G, B, labels, parent_dir + "16")
    read_path(R, G, B, labels, parent_dir + "17")
    read_path(R, G, B, labels, parent_dir + "18")

    return R, G, B, labels


R_data, G_data, B_data, image_data_label = load_data(parent_dir)
out = open("F:/suger_new/images_data/CSV/rgbhsv.csv", 'w', encoding='utf-8', newline='')
csv_write = csv.writer(out)
csv_write.writerow(['r', 'g', 'b', 'label'])
print(R_data)
print(G_data)
print(B_data)
for item in zip(R_data, G_data, B_data, image_data_label):
    item = list(item)
    item[0] = str(item[0])
    item[1] = str(item[1])
    item[2] = str(item[2])
    item[3] = str(item[3])
    item = np.array(item)
    item = item.tolist()

    csv_write.writerow(item)
