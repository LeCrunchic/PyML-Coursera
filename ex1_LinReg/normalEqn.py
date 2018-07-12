import numpy as np

def normal_equation(x, y):
    parameters = np.linalg.pinv(x.T.dot(x)).dot(x.T).dot(y)
    return parameters
    