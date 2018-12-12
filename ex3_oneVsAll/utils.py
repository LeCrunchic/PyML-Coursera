from math import floor, ceil, sqrt
from functools import partial
import pdb

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_bfgs
from numpy import ix_

class OneVsAll:
    def __init__(self, X, Y, num_labels, lambda_):
        self.x = X
        self.y = Y
        self.num_labels = num_labels
        self.lambda_ = lambda_
        self.m, self.n = self.x.shape

    def fit(self):
        all_theta = np.zeros((self.num_labels, self.n + 1))
        x = np.hstack((np.ones((self.m, 1)), self.x))

        initial_theta = np.zeros((self.n + 1, 1))

        print('Training One vs all model...')

        # The addition of one (line 28) is needed because the labels start from 1, python starts counting from 0.
        for i in range(self.num_labels):
            y_train = (self.y == (i + 1)).flatten()
            f = partial(compute_cost, x=x, y=y_train, regularized=True, lambda_=self.lambda_)
            fprime = partial(compute_gradient, x=x, y=y_train, regularized=True, lambda_=self.lambda_)
            i_theta = fmin_bfgs(f, initial_theta.flatten(), fprime, maxiter=100)
            all_theta[i, :] = i_theta

        self.learned_theta = all_theta

    def predict(self):
        self.x = np.hstack((np.ones((self.m, 1)), self.x))

        preds = sigmoid(self.x.dot(self.learned_theta.T))
        argmax = np.argmax(preds, axis=1)

        return argmax

class NeuralNetwork:
    """
         Single hidden layer neural network w/ pre-trained weights
    """
    def __init__(self, x, y, w1, w2):
        self.x = x
        self.y = y
        self.w1 = w1
        self.w2 = w2

    def predict(self, train=True, test_x=None):
        if train:
            x = self.x
        else:
            x = test_x

        x = np.hstack( (np.ones((x.shape[0], 1)), x) )
        Z1 = x.dot(self.w1.T)
        a1 = np.hstack( ( np.ones((x.shape[0], 1)), sigmoid(Z1) ) )
        Z2 = a1.dot(self.w2.T)
        hx = sigmoid(Z2)

        return np.argmax(hx, axis=1)


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
    plt.show(block=False)

def sigmoid(x):
    return 1. / (1. + np.exp(-x))
