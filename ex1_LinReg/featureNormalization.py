import numpy as np

def normalize(x):
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0, ddof=1)
    xNorm = (x - mu) / sigma
    return mu, sigma, xNorm