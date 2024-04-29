import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
import torch
from learn_phi import exp_proj, train_model

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F

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

M = [100]
n_classes = 10
N = 28 * 28
accuracy = []

def rotate_img(image, angle):

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
    return result

train_path = "Dataset/MNIST/train"
train_imgs = os.listdir(train_path)

# sampling matrix
phi = torch.randn(M, N)

images = []
starts = []

for i in range(n_classes):
    id = 0
    for j in range(len(train_imgs)):
        if(id == 0):
            starts.append(len(images))
            id = 1
        if map[train_imgs[j].split("_")[0]] == i:
            images.append(np.array(plt.imread(train_path + "/" + train_imgs[j])).flatten())

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

# M = [10, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375]

for l in range(len(M)):
    
    train_model(phi, optimizer, train, clusters)

    phi = torch.d
    centers = np.zeros((n_classes, M[l]))
    counts = np.zeros(n_classes)

    phi = np.random.randn(M[l], N)


    for i in range(len(train_imgs)):
        counts[map[train_imgs[i].split("_")[0]]] += 1
        centers[map[train_imgs[i].split("_")[0]],:] += phi @ plt.imread(train_path + "/" + train_imgs[i]).flatten()


    for i in range(10):
        centers[i,:] = centers[i,:]/counts[i]


    test_path = "Dataset/MNIST/test"
    test_imgs = os.listdir(test_path)

    x = []
    y = []

    for i in range(len(test_imgs)):
        x.append(rotate_img(plt.imread(test_path + "/" + test_imgs[i]), np.random.randint(0, 360)))
        y.append(map[test_imgs[i].split("_")[0]])

    thetas = np.arange(5, 360, 10)

    preds = []

    for i in range(len(x)):
        pred = 0
        min_dist = np.inf
        min_angle = np.inf

        for j in range(len(thetas)):
            r_img = rotate_img(x[i], -thetas[j])
            meas = phi @ r_img.flatten()

            for k in range(len(centers)):
                if(np.linalg.norm(meas - centers[k,:]) < min_dist):
                    min_dist = np.linalg.norm(meas - centers[k,:])
                    min_angle = thetas[j]
                    pred = k
        preds.append(pred)


    c = 0
    for i in range(len(preds)):
        if(preds[i] == y[i]):
            c += 1

    # print("classification accuracy ->", 100*c/len(preds))
    accuracy.append(100*c/len(preds))
    print(l, "th iteration done")

plt.plot(M, accuracy)
plt.xlabel("No. of meansurements")
plt.ylabel("Classification accuracy")
plt.show()
