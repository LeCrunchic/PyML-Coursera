from math import ceil, floor, sqrt
from functools import partial
from pprint import pprint

import numpy as np
import matplotlib.pyplot as plt
from numpy import ix_

def debug_initialize_weights(outdim, indim):

    W = np.zeros((outdim, indim + 1))
    W = np.sin(np.arange(1, W.size+1)).reshape(W.shape, order='F') / 10

    return W


def check_nn_gradients(lambda_=0):

    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    W1 = debug_initialize_weights(hidden_layer_size, input_layer_size)
    W2 = debug_initialize_weights(num_labels, hidden_layer_size)

    X = debug_initialize_weights(m, input_layer_size - 1)
    Y = 1 + np.arange(1, m+1) % num_labels

    W1 = W1.flatten(order='F')[:, np.newaxis]
    W2 = W2.flatten(order='F')[:, np.newaxis]

    nn_params = np.vstack(( W1, W2 ))

    cost_func = partial(nn_cost_function, input_layer_size=input_layer_size,
                        hidden_layer_size=hidden_layer_size,
                        num_labels=num_labels, x=X, y=Y, lambda_=lambda_)

    J, grad = cost_func(nn_params)
    numgrad = compute_numerical_grad(cost_func, nn_params)

    numgrad_grad = list( zip(numgrad, grad) )
    pprint(numgrad_grad)

    diff = np.linalg.norm(numgrad-grad[:, np.newaxis]) / np.linalg.norm(numgrad+grad[:, np.newaxis])
    import pdb; pdb.set_trace()

    print('difference (ought to be less than 1e-9):', diff)


def compute_numerical_grad(J, theta):

    numgrad = np.zeros(theta.shape, dtype=float)
    perturb = np.zeros(theta.shape, dtype=float)

    e = 1e-4

    for p in range(theta.size):
        perturb[p] = e
        loss1, _ = J(theta - perturb)
        loss2, _ = J(theta + perturb)

        numgrad[p] = (loss2 - loss1) / (2*e)
        perturb[p] = 0

    return numgrad


def rand_init_weights(inL, outL):

    epsinit = sqrt(6)/sqrt(inL + outL)
    W = np.random.randn(inL, outL + 1) * (2 * epsinit) - epsinit
    return W


def sigmoid_gradient(x):

    gradient = sigmoid(x) * (1 - sigmoid(x))
    return gradient


def nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, x, y, lambda_):

    UNTIL_W1 = hidden_layer_size * (input_layer_size + 1)

    W1 = nn_params[:UNTIL_W1].reshape(hidden_layer_size, input_layer_size + 1, order='F')
    W2 = nn_params[UNTIL_W1:].reshape(num_labels, hidden_layer_size + 1, order='F')


    m = x.shape[0]

    x = np.hstack( (np.ones((m, 1)), x) )
    Z1 = x.dot(W1.T)
    a1 = np.hstack( ( np.ones((m, 1)), sigmoid(Z1) ) )
    Z2 = a1.dot(W2.T)
    hx = sigmoid(Z2)

    vectorY = np.zeros((m, num_labels))

    for i in range(m):
        vectorY[i, y[i]-1] = 1

    J = (-1/m) * np.sum(np.sum(vectorY * np.log(hx) + (1 - vectorY) * np.log(1 - hx)))


    w1_no_bs = W1[:, 1:]
    w2_no_bs = W2[:, 1:]
    weights_no_bs = np.hstack((w1_no_bs.flatten(order='F'), w2_no_bs.flatten(order='F')))

    # Regularazion of Cost
    J += (lambda_/(2*m)) * np.sum(weights_no_bs**2)

    # Backpropagation

    dhx = (hx - vectorY)
    d1 = dhx.dot(w2_no_bs) * sigmoid_gradient(Z1)

    D1 = d1.T.dot(x)
    D2 = dhx.T.dot(a1)

    w1_grad = 1/m * D1
    w2_grad = 1/m * D2

    w1_grad = np.hstack((w1_grad[:, [0]], w1_grad[:, 1:] + (lambda_/m) * w1_no_bs))

    w2_grad = np.hstack((w2_grad[:, [0]], w2_grad[:, 1:] + (lambda_/m) * w2_no_bs))

    grad = np.hstack((w1_grad.flatten(order='F'), w2_grad.flatten(order='F')))

    return J, grad


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

            m_idxer = pad + j * (example_height + pad) + np.arange(0, example_height)
            n_idxer = pad + i * (example_width + pad) + np.arange(0, example_width)

            display_array[ix_(m_idxer, n_idxer)] = \
                                  x[current_ex, :].reshape(example_height, example_width, order='F') / max_val

            current_ex += 1

        if current_ex > m:
            break

    plt.imshow(display_array, cmap='cool')
    plt.show(block=False)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
