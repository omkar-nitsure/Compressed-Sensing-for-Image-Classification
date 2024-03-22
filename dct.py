import numpy as np


def DCT(N):

    dct = np.zeros((N, N))

    for u in range(N):
        for v in range(N):
            if u == 0:
                dct[u][v] = np.sqrt(1 / N)
            else:
                dct[u][v] = np.sqrt(2 / N) * np.cos(np.pi * (2 * v + 1) * u / (2 * N))

    return dct
