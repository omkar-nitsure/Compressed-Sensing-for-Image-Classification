import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
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


M = np.arange(10, 150, 10)
sigma = [0, 10, 20, 30, 40]
n_classes = 10
N = 28 * 28
accuracy = np.zeros((len(sigma), len(M)))

def rotate_img(image, angle):

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
    return result

path_train = "Dataset/MNIST/train"
imgs_train = os.listdir(path_train)

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
    x.append(plt.imread(test_path + "/" + test_imgs[i]).flatten())
    y.append(map[test_imgs[i].split("_")[0]])

for l in range(len(M)):

    print("M ->", M[l])

    for s in range(len(sigma)):

        phi = torch.load("models/learnt_phi/phi_" + str(M[l]) + ".pt")
        phi.requires_grad = False
        phi = phi.detach().numpy()

        #uncomment this if you want random normal phi
        # phi = np.random.randn(M[l], N)
        noise = np.random.normal(0, sigma[s], (M[l],))

        centers = np.zeros((n_classes, M[l]))
        for i in range(n_classes - 1):
            centers[i,:] = np.mean((phi @ images[starts[i]:starts[i + 1]].T).T, axis=0)

        centers[9,:] = np.mean((phi @ images[starts[9]:len(images)].T).T, axis=0)

        c = 0
        for i in range(len(x)):
            meas = (phi @ x[i]) + noise
            y_pred = np.argmin(np.linalg.norm(meas - centers, axis=1))
            if(y_pred == y[i]):
                c += 1

        accuracy[s, l] = 100*c/len(x)

print(accuracy)


np.save("plots/GMLC_noisy.npy", accuracy)
plt.plot(M, accuracy[0, :], label="sigma = 0")
plt.plot(M, accuracy[1, :], label="sigma = 10")
plt.plot(M, accuracy[2, :], label="sigma = 20")
plt.plot(M, accuracy[3, :], label="sigma = 30")
plt.plot(M, accuracy[4, :], label="sigma = 40")
plt.legend()   
plt.xlabel("No. of meansurements")
plt.ylabel("Classification accuracy")
plt.title("Variation of classification Accuracy with noise")
plt.savefig("plots/GMLC_acc_noisy.png")