import numpy as np

def leGradientDescent(x, y, learning_rate, iterations, parameters):
    m = x.shape[0]
    for i in range(iterations):
        parameters = parameters - (1/m) * learning_rate * ( (x.dot(parameters) - y).T.dot(x) ).T
    return parameters
