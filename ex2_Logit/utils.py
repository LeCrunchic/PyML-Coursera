import numpy as np

def costFunction(theta, x, y):
    m = y.shape[0]
    J = -(1/m) * np.sum(y * np.log(sigmoid(x.dot(theta))) + (1 - y) * np.log(1 - sigmoid(x.dot(theta))) ) 
    grad = (1/m) * ((sigmoid(x.dot(theta)) - y).T.dot(x)).T
    return J, grad

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
