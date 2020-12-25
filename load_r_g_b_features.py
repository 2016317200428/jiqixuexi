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


image_data_label = []
parent_dir = 'F:/sugar_new/images_data/'
IMAGE_SIZE = 1024
rg_data = []
rb_data = []
gb_data = []

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
    img1 = cv2.imread(path)
    img = resize_image(img1)
    b, g, r = cv2.split(img)
    #img = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    npim = np.zeros((IMAGE_SIZE,IMAGE_SIZE), dtype=np.float)
    sum_rgb = 0
    npim[:] = img[:, :, 0] + img[:, :, 1] + img[:, :, 2]
    rg = np.zeros((1024, 1024))
    rb = np.zeros((1024, 1024))
    gb = np.zeros((1024, 1024))
    for i in range(1024):
        for j in range(1024):
            if npim[i, j] != 0:
                if r[i,j]!=0 and g[i,j] != 0:
                    rg[i, j] = r[i,j] / g[i,j]
                if r[i,j]!=0 and  b[i,j] != 0:
                    rb[i, j] = r[i,j] / b[i,j]
                if g[i,j] !=0 and b[i,j] != 0:
                    gb[i, j] = g[i,j]/b[i,j]
                sum_rgb = sum_rgb + 1

    rg_mean = np.sum(rg[:, :])/sum_rgb
    rb_mean = np.sum(rb[:, :])/sum_rgb
    gb_mean = np.sum(gb[:, :])/sum_rgb

    return rg_mean, rb_mean, gb_mean


def read_path(rg, rb, gb,  labels, path_name):
    for dir_item in os.listdir(path_name):
        label = float(dir_item.split(' ')[0])
        full_path = os.path.abspath(os.path.join(path_name, dir_item))
        rg, rb, gb = compute(full_path)

        rg.append(rg)
        rb.append(rb)
        gb.append(gb)
        labels.append(label)


def load_data(parent_dir):
    rg = []
    rb = []
    gb = []
    labels = []

    read_path(rg, rb, gb, labels, parent_dir + "13")
    read_path(rg, rb, gb, labels, parent_dir + "14")
    read_path(rg, rb, gb, labels, parent_dir + "15")
    read_path(rg, rb, gb, labels, parent_dir + "16")
    read_path(rg, rb, gb, labels, parent_dir + "17")
    read_path(rg, rb, gb, labels, parent_dir + "18")

    return rg, rb, gb, labels


rg_data, rb_data, gb_data, image_data_label = load_data(parent_dir)
out = open("F://sugar_new/images_data/CSV/demo", 'w', encoding='utf-8', newline='')
csv_write = csv.writer(out)
csv_write.writerow(['r/g', 'r/b', 'g/b', 'label'])
print(rg_data)
print(rb_data)
print(gb_data)
for item in zip(rg_data, rb_data, gb_data, image_data_label):
    item = list(item)
    item[0] = str(item[0])
    item[1] = str(item[1])
    item[2] = str(item[2])
    item[3] = str(item[3])
    item = np.array(item)
    item = item.tolist()

    csv_write.writerow(item)
