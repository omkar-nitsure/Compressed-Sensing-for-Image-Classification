import numpy as np
import os
import pickle
import cv2 as cv

map = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}


def unpickle(file):

    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


root_path = "Dataset"

if not os.path.exists(root_path):
    os.makedirs(root_path)

path = "Dataset/CIFAR-10"

if not os.path.exists(path):
    os.makedirs(path)

path = "Dataset/CIFAR-10/train"

if not os.path.exists(path):
    os.makedirs(path)

path = "Dataset/CIFAR-10/validation"

if not os.path.exists(path):
    os.makedirs(path)

path = "Dataset/CIFAR-10/test"

if not os.path.exists(path):
    os.makedirs(path)

x_train = []
x_test = []
x_val = []
y_train = []
y_test = []
y_val = []

path = "Dataset/CIFAR-10/cifar-10-batches-py/"

for i in range(1, 5):

    subset = unpickle(os.path.join(path, f"data_batch_{i}"))
    x_train.append(subset[b"data"])
    y_train.append(subset[b"labels"])

subset = unpickle(os.path.join(path, f"data_batch_5"))
x_val.append(subset[b"data"])
y_val.append(subset[b"labels"])

test_batch = unpickle(os.path.join(path, "test_batch"))
x_test.append(test_batch[b"data"])
y_test.append(test_batch[b"labels"])


x_train = np.concatenate(x_train, axis=0)
x_val = np.concatenate(x_val, axis=0)
x_test = np.concatenate(x_test, axis=0)

x_train = x_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.int32)
x_val = x_val.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.int32)
x_test = x_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.int32)

y_train = np.concatenate(y_train, axis=0)
y_val = np.concatenate(y_val, axis=0)
y_test = np.concatenate(y_test, axis=0)

path = "Dataset/CIFAR-10/train/"
count = np.ones(10, dtype=np.int32)
for i in range(len(x_train)):
    cv.imwrite(
        path + map[y_train[i]] + "_" + str(count[y_train[i]]) + ".png", x_train[i]
    )
    count[y_train[i]] += 1

path = "Dataset/CIFAR-10/validation/"
count = np.ones(10, dtype=np.int32)
for i in range(len(x_val)):
    cv.imwrite(path + map[y_val[i]] + "_" + str(count[y_val[i]]) + ".png", x_val[i])
    count[y_val[i]] += 1

path = "Dataset/CIFAR-10/test/"
count = np.ones(10, dtype=np.int32)
for i in range(len(x_test)):
    cv.imwrite(path + map[y_test[i]] + "_" + str(count[y_test[i]]) + ".png", x_test[i])
    count[y_test[i]] += 1

print("CIFAR-10 dataset loaded successfully!!")
