import os
import numpy as np
import cv2 as cv

data = np.load("mnist.npz")

map = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
}

n_imgs = np.ones(10, dtype=np.int32)


x_train, y_train, x_test, y_test = (
    data["x_train"],
    data["y_train"],
    data["x_test"],
    data["y_test"],
)

root_path = "Dataset"

if not os.path.exists(root_path):
    os.makedirs(root_path)

path = "Dataset/MNIST"

if not os.path.exists(path):
    os.makedirs(path)

path = "Dataset/MNIST/train"

if not os.path.exists(path):
    os.makedirs(path)

path = "Dataset/MNIST/validation"

if not os.path.exists(path):
    os.makedirs(path)

path = "Dataset/MNIST/test"

if not os.path.exists(path):
    os.makedirs(path)


for i in range(len(x_train)):
    if i < 50000:
        path = "Dataset/MNIST/train/"
    else:
        path = "Dataset/MNIST/validation/"
    cv.imwrite(
        path + map[y_train[i]] + "_" + str(n_imgs[y_train[i]]) + ".png", x_train[i]
    )
    n_imgs[y_train[i]] += 1

path = "Dataset/MNIST/test/"
n_imgs = np.ones(10, dtype=np.int32)

for i in range(len(x_test)):
    cv.imwrite(path + map[y_test[i]] + "_" + str(n_imgs[y_test[i]]) + ".png", x_test[i])
    n_imgs[y_test[i]] += 1

print("MNIST dataset loaded successfully!!")
