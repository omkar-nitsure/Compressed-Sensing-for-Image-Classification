import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn

# constants
M = [100]
N = 28 * 28
n_classes = 10

# dictionary map
map = {
    "zero":0,
    "one":1,
    "two":2,
    "three":3,
    "four":4,
    "five":5,
    "six":6,
    "seven":7,
    "eight":8,
    "nine":9
}

# extracting training data for cluster computation
path_train = "Dataset/MNIST/train"
imgs_train = os.listdir(path_train)

# uncomment this part if you want to use learnt sampling matrix
phi = torch.load("models/phi_correct_150.pt")
phi.requires_grad = False
phi = phi.detach().numpy()

# uncomment this if you want random gaussian sampling matrix
# phi = np.random.randn(M, N)


# testing
path = "Dataset/MNIST/test"
imgs = os.listdir(path)
test_image = []
test_label = []

for i in range(n_classes):
    for j in range(len(imgs)):
        if map[imgs[j].split("_")[0]] == i:
            test_image.append(plt.imread(path + "/" + imgs[j]).flatten())
            test_label.append(i)

test_image = np.array(test_image)

images = []
starts = []


for i in range(n_classes):
    id = 0
    for j in range(len(imgs_train)):
        if(id == 0):
            starts.append(len(images))
            id = 1
        if map[imgs_train[j].split("_")[0]] == i:
            images.append(np.array(plt.imread(path_train + "/" + imgs_train[j])).flatten())

images = np.array(images)

for j in range(len(M)):
    clusters = np.zeros((n_classes, M[j]))
    for i in range(n_classes - 1):
        clusters[i,:] = np.mean((phi @ images[starts[i]:starts[i + 1]].T).T, axis=0)

    clusters[9,:] = np.mean((phi @ images[starts[9]:len(images)].T).T, axis=0)


    c = 0
    for i in range(len(test_image)):
        x = phi @ test_image[i]
        y_pred = np.argmin(np.linalg.norm(x - clusters, axis=1))
        if(y_pred == test_label[i]):
            c += 1

    print("with learnt sampling matrix, classification accuracy is ->", 100*c/len(test_image))
