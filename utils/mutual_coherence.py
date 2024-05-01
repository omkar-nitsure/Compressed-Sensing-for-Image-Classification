import numpy as np
import matplotlib.pyplot as plt
import torch


N = 28 * 28
M = np.arange(10, 400, 10)

mu_vals_rand = []
mu_vals_learn = []

for i in range(len(M)):

    

    phi = np.random.randn(M[i], N)
    phi = phi/np.linalg.norm(phi, axis=0)

    mu = np.abs((phi.T @ phi))
    for z in range(mu.shape[0]):
        mu[z,z] = 0

    mu = np.max(mu)
    mu_vals_rand.append(mu)


for i in range(len(M)):

    phi = torch.load("../models/learnt_phi/phi_" + str(M[i]) + ".pt", map_location=torch.device('cpu'))
    phi.requires_grad = False
    phi = phi.detach().numpy()
    phi = phi/np.linalg.norm(phi, axis=0)

    mu = np.abs((phi.T @ phi))

    for z in range(mu.shape[0]):
        mu[z,z] = 0
    mu = np.max(mu)
    mu_vals_learn.append(mu)

plt.plot(M, mu_vals_rand, label="random")
plt.plot(M, mu_vals_learn, label="learnt")
plt.xlabel("No of measurements")
plt.ylabel("Values of Mutual Coherence")
plt.title("variation of Mutual Coherence with No of measurements")
plt.legend()
plt.savefig("../plots/coherence_plot.png")
