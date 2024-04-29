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


M = [100]
sigma = [10, 20]
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
    x.append(rotate_img(plt.imread(test_path + "/" + test_imgs[i]), np.random.randint(0, 360)))
    y.append(map[test_imgs[i].split("_")[0]])


for l in range(len(M)):

    print("M ->", M[l])

    for s in range(len(sigma)):
            
        centers = np.zeros((n_classes, M[l]))

        phi = torch.load("/home/ojas/AIP_Project/models/phi_correct_150.pt")
        phi.requires_grad = False
        phi = phi.detach().numpy()
        noise = np.random.normal(0, sigma[s], (M[l],))

        centers = np.zeros((n_classes, M[l]))
        for i in range(n_classes - 1):
            centers[i,:] = np.mean((phi @ images[starts[i]:starts[i + 1]].T).T, axis=0)

        centers[9,:] = np.mean((phi @ images[starts[9]:len(images)].T).T, axis=0)


        thetas = np.arange(5, 360, 10)

        preds = []

        for i in range(len(x)):
            pred = 0
            min_dist = np.inf
            min_angle = np.inf

            for j in range(len(thetas)):
                r_img = rotate_img(x[i], -thetas[j]).flatten()
                meas = (phi @ r_img) + noise

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
        accuracy[s, l] = 100*c/len(preds)

print(accuracy)


np.save("values_32/GMLC_with_noisy_meas.npy", accuracy)
plt.plot(M, accuracy[0, :], label="sigma = 10")
plt.plot(M, accuracy[1, :], label="sigma = 20")
# plt.plot(M, accuracy[2, :], label="sigma = 0.05")
# plt.plot(M, accuracy[3, :], label="sigma = 0.1")
plt.legend()   
plt.xlabel("No. of meansurements")
plt.ylabel("Classification accuracy")
plt.title("Variation of classification Accuracy with noise")
plt.savefig("plots/plots_32/acc_vsnoisy_learnt.png")