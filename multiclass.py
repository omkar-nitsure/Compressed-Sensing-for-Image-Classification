import numpy as np
import cv2 as cv
import os

classes = {0: "zero", 1: "one", 4: "four"}

root_path = "Dataset"

if not os.path.exists(root_path):
    os.makedirs(root_path)

path = "Dataset/Multiclass_MNIST"

if not os.path.exists(path):
    os.makedirs(path)

path = "Dataset/Multiclass_MNIST/train"

if not os.path.exists(path):
    os.makedirs(path)

path = "Dataset/Multiclass_MNIST/validation"

if not os.path.exists(path):
    os.makedirs(path)

path = "Dataset/Multiclass_MNIST/test"

if not os.path.exists(path):
    os.makedirs(path)

path = "Dataset/MNIST/train"
dest = "Dataset/Multiclass_MNIST/train"

imgs = os.listdir(path)

for j in classes.keys():

    for i in range(len(imgs)):
        if imgs[i].split("_")[0] == classes[j]:
            img = cv.imread(path + "/" + imgs[i])
            cv.imwrite(dest + "/" + imgs[i], img)

path = "Dataset/MNIST/validation"
dest = "Dataset/Multiclass_MNIST/validation"

imgs = os.listdir(path)

for j in classes.keys():

    for i in range(len(imgs)):
        if imgs[i].split("_")[0] == classes[j]:
            img = cv.imread(path + "/" + imgs[i])
            cv.imwrite(dest + "/" + imgs[i], img)

path = "Dataset/MNIST/test"
dest = "Dataset/Multiclass_MNIST/test"

imgs = os.listdir(path)

for j in classes.keys():

    for i in range(len(imgs)):
        if imgs[i].split("_")[0] == classes[j]:
            img = cv.imread(path + "/" + imgs[i])
            cv.imwrite(dest + "/" + imgs[i], img)
