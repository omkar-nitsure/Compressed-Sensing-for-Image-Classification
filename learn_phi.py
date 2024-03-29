import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim


class Projection(nn.Module):
    def __init__(self, M, N):
        super(Projection, self).__init__()

        self.phi = nn.Parameter(torch.randn(M, N))

    def forward(self, X):
        return (self.phi) @ X

    def get_phi(self):
        return self.phi


path = "Dataset/MNIST/train"

imgs = os.listdir(path)

three = []
five = []
eight = []

for i in range(len(imgs)):

    if imgs[i].split("_")[0] == "three":
        three.append(np.array(plt.imread(path + "/" + imgs[i]).flatten()))
    elif imgs[i].split("_")[0] == "five":
        five.append(np.array(plt.imread(path + "/" + imgs[i]).flatten()))
    elif imgs[i].split("_")[0] == "eight":
        eight.append(np.array(plt.imread(path + "/" + imgs[i]).flatten()))

three = torch.tensor(np.array(three)).float()
five = torch.tensor(np.array(five)).float()
eight = torch.tensor(np.array(eight)).float()

three = three[0 : five.shape[0]]
eight = eight[0 : five.shape[0]]

train = data.TensorDataset(three, five, eight)
train = data.DataLoader(train, batch_size=32, shuffle=True)

M = 60
N = 28 * 28


def loss_fn(t_mean, f_mean, e_mean):
    loss = (
        M
        * N
        / (
            torch.sum((t_mean - f_mean) ** 2)
            + torch.sum((t_mean - e_mean) ** 2)
            + torch.sum((e_mean - f_mean) ** 2)
        )
    )

    return loss


model = Projection(M, N)
n_epochs = 3
optimizer = optim.SGD(model.parameters(), lr=0.01)


def train_model(model, optimizer, train):

    model.train(True)

    for epoch in range(n_epochs):

        e_loss = 0

        for _, x in enumerate(train):

            t, f, e = x[0], x[1], x[2]

            t_out = model(t.T).T
            f_out = model(f.T).T
            e_out = model(e.T).T

            t_mean = torch.mean(t_out, 0)
            f_mean = torch.mean(f_out, 0)
            e_mean = torch.mean(e_out, 0)

            loss = loss_fn(t_mean, f_mean, e_mean)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            e_loss += loss.item()

        print(epoch, "->", np.round(e_loss, 2))


train_model(model, optimizer, train)

torch.save(model, "models/phi.pt")
