from math import floor, ceil, sqrt
from functools import partial
import pdb

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_bfgs
from numpy import ix_

def one_vs_all(x, y, num_labels, lambda_):
    m, n = x.shape
    all_theta = np.zeros((num_labels, n + 1))
    x = np.hstack((np.ones((m, 1)), x))

    initial_theta = np.zeros((n + 1, 1))

    for i in range(num_labels):
        y_train = (y == i).flatten()
        f = partial(compute_cost, x=x, y=y_train, regularized=True, lambda_=lambda_)
        fprime = partial(compute_gradient, x=x, y=y_train, regularized=True, lambda_=lambda_)
        i_theta = fmin_bfgs(f, initial_theta.flatten(), fprime, maxiter=50)
        all_theta[i, :] = i_theta

    return all_theta

def compute_cost(theta, x, y, regularized=False, lambda_=None):
    m = y.shape[0]

    J = -(1./m) * np.sum(y * np.log(sigmoid(x.dot(theta))) + (1. - y) * np.log(1. - sigmoid(x.dot(theta))) )

    if regularized:
        # notice that we do not regularize the bias term (we start from the second element in theta)
        reg_term = (lambda_ / (2*m)) * np.sum(np.square(theta[1:]))
        J += reg_term

    return J

def compute_gradient(theta, x, y, regularized=False, lambda_=None):
    m = y.shape[0]
    grad = (1./m) * ((sigmoid(x.dot(theta)) - y).T.dot(x)).T

    if regularized:
        reg_term = (lambda_ / m) * theta[1:]

        # Remember each row of x is a training example 
        if grad.ndim > 1:
            # This executes for the actual training, the first row of grad are the gradients of the bias 
            pdb.set_trace()
            grad = np.vstack((grad[0, :], grad[1:, :] + reg_term[:, np.newaxis]))
        else:
            # This executes for the test, grad is flat here
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

            max_val = np.max(np.abs(x[current_ex, :]))

            m_idxer = pad + (j-1) * (example_height + pad) + np.arange(0, example_height)
            n_idxer = pad + (i-1) * (example_width + pad) + np.arange(0, example_width)

            display_array[ix_(m_idxer, n_idxer)] = \
                                  x[current_ex, :].reshape(example_height, example_width, order='F') / max_val

            current_ex += 1

        if current_ex > m:
            break

    plt.imshow(display_array, cmap='gray_r')
    plt.show()

def sigmoid(x):
    return 1. / (1. + np.exp(-x))
