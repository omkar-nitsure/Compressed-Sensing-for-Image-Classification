import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn

# constants
M = 100
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
path = "Dataset/MNIST/train"
imgs = os.listdir(path)

# uncomment this part if you want to use learnt sampling matrix
phi = torch.load("models/learnt_phi/phi_150.pt")
phi.requires_grad = False
phi = phi.detach().numpy()

# uncomment this if you want random gaussian sampling matrix
# phi = np.random.randn(M, N)

images = []
starts = []

for i in range(n_classes):
    id = 0
    for j in range(len(imgs)):
        if(id == 0):
            starts.append(len(images))
            id = 1
        if map[imgs[j].split("_")[0]] == i:
            images.append(np.array(plt.imread(path + "/" + imgs[j])).flatten())

images = np.array(images)

centers = np.zeros((n_classes, M))
for i in range(n_classes - 1):
    centers[i,:] = np.mean((phi @ images[starts[i]:starts[i + 1]].T).T, axis=0)

centers[9,:] = np.mean((phi @ images[starts[9]:len(images)].T).T, axis=0)

del images, starts


# testing
path = "Dataset/MNIST/test"
imgs = os.listdir(path)
x = []
y = []

for i in range(n_classes):
    for j in range(len(imgs)):
        if map[imgs[j].split("_")[0]] == i:
            x.append(phi @ plt.imread(path + "/" + imgs[j]).flatten())
            y.append(i)

x = np.array(x)
c = 0
for i in range(len(x)):
    y_pred = np.argmin(np.linalg.norm(x[i] - centers, axis=1))
    if(y_pred == y[i]):
        c += 1

print(100*c/len(x))


# experiment results
# with learnt sampling matrix, classification accuracy is -> 86.23 %
# with random gaussian sampling matrix, classfication accuracy is -> 76.54%