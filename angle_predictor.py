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

def rotate_img(image, angle):

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
    return result

train_path = "Dataset/MNIST/train"
train_imgs = os.listdir(train_path)

test_path = "Dataset/MNIST/test"
test_imgs = os.listdir(test_path)

x = []
y = []

for i in range(len(test_imgs)):
    x.append(rotate_img(plt.imread(test_path + "/" + test_imgs[i]), np.random.randint(0, 360)))
    y.append(map[test_imgs[i].split("_")[0]])


M = np.arange(10, 400, 10)
n_classes = 10
N = 28 * 28
accuracy_rand = []
acc_learn = []

for l in range(len(M)):
        
    centers = np.zeros((n_classes, M[l]))
    counts = np.zeros(n_classes)

    phi = np.random.randn(M[l], N)


    for i in range(len(train_imgs)):
        counts[map[train_imgs[i].split("_")[0]]] += 1
        centers[map[train_imgs[i].split("_")[0]],:] += phi @ plt.imread(train_path + "/" + train_imgs[i]).flatten()


    for i in range(10):
        centers[i,:] = centers[i,:]/counts[i]


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


for l in range(len(M)):
        
    centers = np.zeros((n_classes, M[l]))
    counts = np.zeros(n_classes)

    phi = torch.load("models/phi_batch_32/phi_" + str(M[l]) + ".pt", map_location=torch.device('cpu'))
    phi.requires_grad = False
    phi = phi.detach().numpy()


    for i in range(len(train_imgs)):
        counts[map[train_imgs[i].split("_")[0]]] += 1
        centers[map[train_imgs[i].split("_")[0]],:] += phi @ plt.imread(train_path + "/" + train_imgs[i]).flatten()


    for i in range(10):
        centers[i,:] = centers[i,:]/counts[i]


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
    acc_learn.append(100*c/len(preds))
    print(l, "th iteration done")

accuracy_rand = np.array(accuracy_rand)
acc_learn = np.array(acc_learn)
np.save("values_32/acc_rand.npy", accuracy_rand)
np.save("values_32/acc_learnt.npy", acc_learn)
plt.plot(M, accuracy_rand, label="random phi")
plt.plot(M, acc_learn, label="learnt phi")
plt.xlabel("No. of meansurements")
plt.ylabel("Classification accuracy")
plt.title("Accuracy Vs No of measurements for random and learnt phi")
plt.legend()
plt.savefig("plots/plots_32/classification_acc_vs_measurements.png")