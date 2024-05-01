import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from sklearn.manifold import TSNE

# constants
M = np.arange(10, 400, 20)
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

phi = torch.load("models/learnt_phi/phi_" + str(390) + ".pt")
phi.requires_grad = False
phi = phi.detach().numpy()

# testing
path = "Dataset/MNIST/test"
imgs = os.listdir(path)
test_image = []
test_label = []
first = []

for i in range(n_classes):
    is_start = 0
    for j in range(len(imgs)):
        if map[imgs[j].split("_")[0]] == i:
            if(is_start == 0):
                first.append(len(test_image))
                is_start = 1
            test_image.append(plt.imread(path + "/" + imgs[j]).flatten())
            test_label.append(i)

test_image = np.array(test_image)

tsne = TSNE(n_components=2, random_state=42)

repr = tsne.fit_transform(test_image)


plt.scatter(repr[first[0] : first[1], 0], repr[first[0] : first[1], 1], label="0")
plt.scatter(repr[first[1] : first[2], 0], repr[first[1] : first[2], 1], label="1")
plt.scatter(repr[first[2] : first[3], 0], repr[first[2] : first[3], 1], label="2")
plt.scatter(repr[first[3] : first[4], 0], repr[first[3] : first[4], 1], label="3")
plt.scatter(repr[first[4] : first[5], 0], repr[first[4] : first[5], 1], label="4")
plt.scatter(repr[first[5] : first[6], 0], repr[first[5] : first[6], 1], label="5")
plt.scatter(repr[first[6] : first[7], 0], repr[first[6] : first[7], 1], label="6")
plt.scatter(repr[first[7] : first[8], 0], repr[first[7] : first[8], 1], label="7")
plt.scatter(repr[first[8] : first[9], 0], repr[first[8] : first[9], 1], label="8")
plt.scatter(repr[first[9] :, 0], repr[first[9] :, 1], label="9")
plt.title("vector of image pixels")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.savefig("image_clusters.png")
