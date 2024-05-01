import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn

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

acc_learnt = []
acc_rand = []

# extracting training data for cluster computation
path_train = "Dataset/MNIST/train"
imgs_train = os.listdir(path_train)


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

    phi = torch.load("models/learnt_phi/phi_" + str(M[j]) + ".pt")
    phi.requires_grad = False
    phi = phi.detach().numpy()
    centers = np.zeros((n_classes, M[j]))
    for i in range(n_classes - 1):
        centers[i,:] = np.mean((phi @ images[starts[i]:starts[i + 1]].T).T, axis=0)

    centers[9,:] = np.mean((phi @ images[starts[9]:len(images)].T).T, axis=0)


    c = 0
    for i in range(len(test_image)):
        x = phi @ test_image[i]
        y_pred = np.argmin(np.linalg.norm(x - centers, axis=1))
        if(y_pred == test_label[i]):
            c += 1

    acc_learnt.append(100*c/len(test_image))

    print("with learnt sampling matrix, classification accuracy is ->", 100*c/len(test_image))




for j in range(len(M)):

    phi = np.random.randn(M[j], N)
    centers = np.zeros((n_classes, M[j]))
    for i in range(n_classes - 1):
        centers[i,:] = np.mean((phi @ images[starts[i]:starts[i + 1]].T).T, axis=0)

    centers[9,:] = np.mean((phi @ images[starts[9]:len(images)].T).T, axis=0)


    c = 0
    for i in range(len(test_image)):
        x = phi @ test_image[i]
        y_pred = np.argmin(np.linalg.norm(x - centers, axis=1))
        if(y_pred == test_label[i]):
            c += 1

    acc_rand.append(100*c/len(test_image))

    print("with random sampling matrix, classification accuracy is ->", 100*c/len(test_image))

np.save("plots/acc_rand.npy", np.array(acc_rand))
np.save("plots/acc_learnt.npy", np.array(acc_learnt))

plt.plot(M, acc_rand, label="random phi")
plt.plot(M, acc_learnt, label="learnt phi")
plt.ylabel("classification accuracy")
plt.xlabel("No of measurements")
plt.legend()
plt.title("Classification accuracy Vs No of measurements")
plt.savefig("plots/classification_acc.png")
