import numpy as np
import matplotlib.pyplot as plt
import torch


N = 28 * 28
M = np.arange(10, 400, 10)

mu_vals_rand = []
mu_vals_learn = []

for i in range(len(M)):

    phi = np.random.randn(M[i], N)
    mu = np.max(np.abs((phi.T @ phi)/(np.linalg.norm(phi.T, axis=1)*np.linalg.norm(phi.T, axis=1))))
    mu_vals_rand.append(mu)


for i in range(len(M)):

    phi = torch.load("models/phi_models/phi_" + str(M[i]) + ".pt", map_location=torch.device('cpu'))
    phi.requires_grad = False
    phi = phi.detach().numpy()
    mu = np.max(np.abs((phi.T @ phi)/(np.linalg.norm(phi.T, axis=1)*np.linalg.norm(phi.T, axis=1))))
    mu_vals_learn.append(mu)


plt.plot(M[:10], mu_vals_rand[:10], label="random")

plt.plot(M[:10], mu_vals_learn[:10], label="learnt")
plt.xlabel("No of measurements")
plt.ylabel("Values of Mutual Coherence")
plt.title("variation of Mutual Coherence with No of measurements")
plt.legend()
plt.savefig("coherence_plot.png")
