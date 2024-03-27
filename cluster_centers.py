import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

M = np.arange(2, 62)
N = 28 * 28
y1 = []
y2 = []

path = "Dataset/Multiclass_MNIST/train"

imgs = os.listdir(path)

four = []
one = []
zero = []

for i in range(len(imgs)):
    if imgs[i].split("_")[0] == "four":
        four.append(np.mean(plt.imread(path + "/" + imgs[i]), 2).flatten())
    elif imgs[i].split("_")[0] == "zero":
        zero.append(np.mean(plt.imread(path + "/" + imgs[i]), 2).flatten())
    else:
        one.append(np.mean(plt.imread(path + "/" + imgs[i]), 2).flatten())

four = np.array(four)
one = np.array(one)
zero = np.array(zero)

for i in range(len(M)):
    phi = np.random.randn(M[i], N)

    z_mean = np.mean(phi @ zero.T, 1)
    f_mean = np.mean(phi @ four.T, 1)
    o_mean = np.mean(phi @ one.T, 1)

    dists = []

    dists.append(np.sqrt(np.sum((z_mean - o_mean) ** 2)))
    dists.append(np.sqrt(np.sum((z_mean - f_mean) ** 2)))
    dists.append(np.sqrt(np.sum((f_mean - o_mean) ** 2)))

    dists = np.array(dists)

    y1.append(np.mean(dists))

    y2.append(np.min(dists))


plt.plot(M, y1, label="mean")
plt.plot(M, y2, label="min")
plt.xlabel("No of measurements (M)")
plt.ylabel("distance")
plt.title("variation of distance between manifold centers with M")
plt.legend()
plt.savefig("manifold_dists.png")
