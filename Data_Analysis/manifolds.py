import os
import numpy as np
import matplotlib.pyplot as plt
from dct import DCT
import cv2 as cv
from sklearn.manifold import TSNE

path = "Dataset/MNIST/validation"
img_names = os.listdir(path)
imgs = np.zeros((len(img_names), 28, 28))

map_MNIST = {
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

id = 0
first = np.zeros(10, dtype=np.int32)
y = []

for j in range(10):
    first[j] = id
    for i in range(len(img_names)):
        if img_names[i].split("_")[0] == map_MNIST[j]:
            imgs[id] = np.mean(cv.imread(path + "/" + img_names[i]), axis=2)
            y.append(j)
            id += 1

y = np.array(y)


dct_manifold = np.zeros((len(imgs), 28 * 28))

dct_coeffs = DCT(28)

for i in range(len(imgs)):
    dct_mat = dct_coeffs @ imgs[i] @ dct_coeffs.T
    dct_manifold[i] = dct_mat.flatten()

dct_manifold = np.round(dct_manifold, 3)

tsne = TSNE(n_components=2, random_state=42)

repr = tsne.fit_transform(dct_manifold)


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
plt.title("DCT of the MNIST classes")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.savefig("DCT_visualization_MNIST.png")


dct_max = np.max(dct_coeffs)

for i in range(dct_coeffs.shape[0]):
    for j in range(dct_coeffs.shape[0]):
        dct_coeffs[i][j] = int((255 * dct_coeffs[i][j]) / dct_max)

cv.imwrite("DCT_matrix_28.png", dct_coeffs)
