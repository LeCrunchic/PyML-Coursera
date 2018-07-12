import numpy as np
from cost import computeCost

def leGradientDescent(x, y, learning_rate, iterations, parameters, multi=False):
    m = x.shape[0]

    if(multi):
        J_history = np.zeros((iterations, 1))
    
    for i in range(iterations):
        parameters = parameters - (1/m) * learning_rate * ( (x.dot(parameters) - y).T.dot(x) ).T
        
        if(multi):
            J_history[i] = computeCost(x, y, parameters)
    
    if(multi):
        return parameters, J_history
    else:
        return parameters
