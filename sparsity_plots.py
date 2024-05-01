import numpy as np
import matplotlib.pyplot as plt

K = [100, 150, 200, 250, 300, 350, 400, 450, 500]
accuracy_rand = [59.14, 51.98, 40.0, 44.88, 49.98, 49.61, 55.99, 47.69, 53.2]

plt.plot(K, accuracy_rand)
plt.xlabel("Sparsity")
plt.ylabel("Classification Accuracy")
plt.title("Classification Accuracy vs Sparsity for fixed number of measurements")
plt.show()