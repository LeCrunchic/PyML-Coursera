import numpy as np

def computeCost(x, y, parameters):
    num_trn = x.shape[0]
    cost = np.sum( np.power(x.dot(parameters) - y, 2) ) / (2*num_trn)
    return cost

