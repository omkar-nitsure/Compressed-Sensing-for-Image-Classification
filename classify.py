import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn


N = 28 * 28

path = "Dataset/MNIST/train"
imgs = os.listdir(path)

three = []
five = []
eight = []

for i in range(len(imgs)):
    if imgs[i].split("_")[0] == "zero":
        three.append(plt.imread(path + "/" + imgs[i]).flatten())
    elif imgs[i].split("_")[0] == "one":
        five.append(plt.imread(path + "/" + imgs[i]).flatten())
    elif imgs[i].split("_")[0] == "four":
        eight.append(plt.imread(path + "/" + imgs[i]).flatten())

three = np.array(three)
five = np.array(five)
eight = np.array(eight)

M = np.arange(0, 61, 2)

acc = []

path = "Dataset/MNIST/test"
imgs = os.listdir(path)

x = []
y = []

for i in range(len(imgs)):
    if imgs[i].split("_")[0] == "zero":
        x.append(plt.imread(path + "/" + imgs[i]).flatten())
        y.append(0)
    elif imgs[i].split("_")[0] == "one":
        x.append(plt.imread(path + "/" + imgs[i]).flatten())
        y.append(1)
    elif imgs[i].split("_")[0] == "four":
        x.append(plt.imread(path + "/" + imgs[i]).flatten())
        y.append(2)

x = np.array(x)

for i in range(len(M)):

    phi = np.random.randn(M[i], N)

    t_mean = np.mean(phi @ three.T, 1)
    f_mean = np.mean(phi @ five.T, 1)
    e_mean = np.mean(phi @ eight.T, 1)

    
    measurements = (phi @ x.T).T
    preds = []

    for i in range(len(x)):

        min_id = 0
        min = 0

        preds.append(
            np.argmin(
                [
                    np.linalg.norm(t_mean - measurements[i]),
                    np.linalg.norm(f_mean - measurements[i]),
                    np.linalg.norm(e_mean - measurements[i]),
                ]
            )
        )

    c = 0

    for i in range(len(preds)):
        if preds[i] == y[i]:
            c += 1

    acc.append(np.round(100 * c / len(preds), 2))

print(acc)
plt.plot(M, acc)
plt.xlabel("Number of Measurements")
plt.ylabel("accuracy")
plt.title("Classification accuracy Vs No of measurements")
plt.savefig("classify_accuracy.png")

# for 60 measurements we get the following accuracies for classification
# accuracy for three classes namely Three, Five, Eight is 81.92
# accuracy for three classes namely Zero, One, Four is 97.03
