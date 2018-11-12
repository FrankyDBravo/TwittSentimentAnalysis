from scipy.sparse import *
import numpy as np
import pickle
import random

def glove_emb(cooc,epochs=10,embedding_dim=20):
    
    xs = np.random.normal(size=(cooc.shape[0], embedding_dim))
    ys = np.random.normal(size=(cooc.shape[1], embedding_dim))

    eta = 0.001
    alpha = 3 / 4
    nmax=100

    for epoch in range(epochs):
        print("epoch {}".format(epoch))
        for i, j, n in zip(cooc.row, cooc.col, cooc.data):
            logn = np.log(n+1)
            fn = min(1.0, (n / nmax) ** alpha)
            x, y = xs[i, :], ys[j, :]
            scale = 2 * eta * fn * (logn - np.dot(x, y))
            xs[i, :] += scale * y
            ys[j, :] += scale * x

    return (xs)