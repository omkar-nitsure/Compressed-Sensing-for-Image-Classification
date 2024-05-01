import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
import torch
from dct import DCT
import torch


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

def rotate_img(image, angle):

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
    return result

train_path = "Dataset/MNIST/train"
train_imgs = os.listdir(train_path)

path_train = "Dataset/MNIST/train"
imgs_train = os.listdir(path_train)
n_classes = 10
starts = []
images = []

for i in range(n_classes):
    id = 0
    for j in range(len(imgs_train)):
        if(id == 0):
            starts.append(len(images))
            id = 1
        if map[imgs_train[j].split("_")[0]] == i:
            images.append(np.array(plt.imread(path_train + "/" + imgs_train[j])).flatten())

images = np.array(images)

test_path = "Dataset/MNIST/test"
test_imgs = os.listdir(test_path)

x = []
y = []

for i in range(len(test_imgs)):
    x.append(rotate_img(plt.imread(test_path + "/" + test_imgs[i]), np.random.randint(0, 360)))
    y.append(map[test_imgs[i].split("_")[0]])


# M = [10, 25, 50, 60, 100, 150, 200, 250, 300, 350, 400, 450, 500]
M = 200

N = 28 * 28
accuracy_rand = []
K = [100, 150, 200, 250, 300, 350, 400, 450, 500]
dct = DCT(28*28)


def get_sparse_rep(img, dct, K):

    result_img = np.zeros(img.shape)

    for i in range(len(img)):

        img_coeffs = dct @ img[i]

        res = sorted(range(len(img_coeffs)), key = lambda sub: img_coeffs[sub])[-K:]

        sparse_coeffs = np.zeros(len(img_coeffs))

        for j in range(len(res)):
            sparse_coeffs[res[j]] = img_coeffs[res[j]]

        result_img[i, :] = dct.T @ sparse_coeffs

    return result_img

for l in range(len(K)):

    phi = np.random.randn(M, N)

    centers = np.zeros((n_classes, M))
    for i in range(n_classes - 1):
        centers[i,:] = np.mean((phi @ get_sparse_rep(images[starts[i]:starts[i + 1]], dct, K[l]).T).T, axis=0)

    centers[9,:] = np.mean((phi @ get_sparse_rep(images[starts[9]:len(images)], dct, K[l]).T).T, axis=0)



    thetas = np.arange(5, 360, 10)

    preds = []

    for i in range(len(x)):
        pred = 0
        min_dist = np.inf
        min_angle = np.inf

        for j in range(len(thetas)):
            r_img = rotate_img(x[i], -thetas[j]).flatten()

            meas = phi @ r_img


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
    accuracy_rand.append(100*c/len(preds))
    print(l, "th iteration done")

print(accuracy_rand)

accuracy_rand = np.array(accuracy_rand)

plt.plot(K, accuracy_rand)

plt.xlabel("No. of meansurements")
plt.ylabel("Classification accuracy")
plt.title("Accuracy Vs No of measurements while imposing sparsity on image")
plt.legend()
plt.show()