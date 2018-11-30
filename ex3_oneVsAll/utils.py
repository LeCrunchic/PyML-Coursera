from math import floor, ceil, sqrt

import numpy as np
import matplotlib.pyplot as plt
from numpy import ix_


def compute_cost(theta, x, y, regularized=False, lambda_=None):
    m = y.shape[0]
    J = -(1./m) * np.sum(y * np.log(sigmoid(x.dot(theta))) + (1. - y) * np.log(1. - sigmoid(x.dot(theta))) )

    if regularized:
        # notice that we do not regularize the bias term (we start from the second element in theta)
        reg_term = lambda_ / (2*m) * np.sum(np.square(theta[1:]))
        J += reg_term

    return J

def compute_gradient(theta, x, y, regularized=False, lambda_=None):
    m = y.shape[0]
    grad = (1./m) * ((sigmoid(x.dot(theta)) - y).T.dot(x)).T

    import pdb; pdb.set_trace()
    if regularized:
        reg_term = (lambda_ / m) * theta[1:]
        grad = np.hstack((grad[0], grad[1:] + reg_term))
    return grad

def display_data(x, example_width=None):
    if not example_width:
        example_width = floor(sqrt(x.shape[1]))

    m, n = x.shape
    example_height = int(n / example_width)
    example_width = int(example_width)

    display_rows = floor(sqrt(m))
    display_cols = ceil(m / display_rows)
    pad = 1

    display_array = -np.ones((pad + display_rows * (example_height + pad),
                             pad + display_cols * (example_width + pad)))

    current_ex = 0
    for j in range(display_rows):
        for i in range(display_cols):
            if current_ex > m:
                break

            #max_val = np.max(np.abs(x[current_ex, :]))

            m_idxer = pad + (j-1) * (example_height + pad) + np.arange(0, example_height)
            n_idxer = pad + (i-1) * (example_width + pad) + np.arange(0, example_width)

            display_array[ix_(m_idxer, n_idxer)] = \
                                  x[current_ex, :].reshape(example_height, example_width, order='F') #/ max_val

            current_ex += 1

        if current_ex > m:
            break

    plt.imshow(display_array, cmap='gray_r')
    plt.show()

def sigmoid(x):
    return 1. / (1. + np.exp(-x))
