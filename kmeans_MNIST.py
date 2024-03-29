import numpy as np
from sklearn.cluster import KMeans
import pickle
import os
import statistics

repr = np.load("MNIST_test.npz")

repr = repr["arr_0"]

y_true = np.load("test_labels.npy")


# K = 10
# kmeans_model = KMeans(n_clusters=K, max_iter=1000)
# kmeans_model.fit(repr)

# pickle.dump(kmeans_model, open("kmeans.sav", 'wb'))

kmeans_model = pickle.load(open("kmeans.sav", "rb"))

y_model = []

for i in range(len(repr)):
    min = np.inf
    min_id = 0
    for j in range(10):
        dist = np.sqrt(np.sum(kmeans_model.cluster_centers_[j] - repr[i]) ** 2)
        if dist < min:
            min = dist
            min_id = j
    y_model.append(min_id)


for i in range(10):
    m = []
    for j in range(len(y_true)):
        if y_true[j] == i:
            m.append(y_model[j])

kmeans_lab = [9, 4, 8, 7, 1, 6, 3, 5, 0, 2]

c = 0

for i in range(len(y_true)):
    if y_model[i] == kmeans_lab[y_true[i]]:
        c += 1

print("accuracy on MNIST with Kmeans", 100 * c / len(y_true))


# accuracy on test = 19.58 %
# accuracy on train = 49 %
