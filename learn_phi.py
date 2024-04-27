import numpy as np
import matplotlib.pyplot as plt
import os
import cv2 as cv

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F

# constants
M = 60
N = 28*28
n_classes = 10
sigma = 10

# dictionary used for mapping the class labels
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

def exp_proj(x, clusters):
    Kc = torch.zeros(len(x), n_classes)

    for i in range(len(x)):
        diff = x[i] - clusters
        Kc[i] = (- diff**2).mean(1).div(2 * sigma**2).exp()

    return Kc.float()
    

# computing first set of cluster centers
path = "Dataset/MNIST/train"
imgs = os.listdir(path)


# sampling matrix
phi = torch.randn(M, N)


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

images = torch.tensor(np.array(images)).float()



clusters = torch.zeros((n_classes, M)).float()
for i in range(n_classes - 1):
    clusters[i,:] = torch.mean((phi @ images[starts[i]:starts[i + 1]].T).T, axis=0)

clusters[9,:] = torch.mean((phi @ images[starts[9]:len(images)].T).T, axis=0)


y = torch.zeros(len(images))
for i in range(n_classes - 1):
    y[starts[i]:starts[i + 1]] = i
y[starts[9]:len(images)] = 9

y = F.one_hot(y.to(torch.int64), num_classes=10).float()



train = data.TensorDataset(images, y)
train = data.DataLoader(train, batch_size=32, shuffle=True)

optimizer = optim.Adam([phi], lr=0.01)
n_epochs = 5


def train_model(phi, optimizer, train, clusters):

    phi.requires_grad_()

    for epoch in range(n_epochs):

        e_loss = 0

        for _, x in enumerate(train):

            x_, y = x[0], x[1]

            loss = F.binary_cross_entropy(exp_proj((phi @ x_.T).T, clusters), y)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            e_loss += loss.item()

        print(epoch, "->", np.round(e_loss, 2))

        # recomputing clusters after every epoch
        with torch.no_grad():
            clusters = torch.zeros((n_classes, M)).float()
            for i in range(n_classes - 1):
                clusters[i,:] = torch.mean((phi @ images[starts[i]:starts[i + 1]].T).T, axis=0)

            clusters[9,:] = torch.mean((phi @ images[starts[9]:len(images)].T).T, axis=0)


train_model(phi, optimizer, train, clusters)

with torch.no_grad():
    torch.save(phi, "models/phi_correct.pt")