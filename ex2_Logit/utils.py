import numpy as np
import matplotlib.pyplot as plt

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

    if regularized:
        reg_term = (lambda_ / m) * theta[1:]
        grad = np.hstack((grad[0], grad[1:] + reg_term))
    return grad

def plot_data(x, y, xlabel, ylabel, legend):

    posi = y == 1
    neg = y == 0

    plt.scatter(x[posi, 0], x[posi, 1], marker='+')
    plt.scatter(x[neg, 0], x[neg, 1], c='red' , marker='x')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(legend)
    plt.show()

def plot_bounded_data(x, y, theta):

    posi = y == 1
    neg = y == 0
    
    fig, ax = plt.subplots()

    # Boundaries
    if x.shape[1] <= 3:
        x_plot = np.array([min(x[:, 1]), max(x[:, 1])])
        y_plot =  (-1. / theta[1]) * (theta[2] * x_plot + theta[0])

        ax.scatter(x[posi, 1], x[posi, 2], marker='+', label='Score: First exam.')
        ax.scatter(x[neg, 1], x[neg, 2], c='r' , marker='x', label='Score: Second exam.')
        ax.plot(x_plot, y_plot, linestyle='--', c='g', label='Decision Boundary')

        ax.xlabel('Score: First exam')
        ax.ylabel('Score: Second exam')
        ax.legend()

    else:
       x_plot = np.linspace(-0.75, 1.)
       y_plot = np.linspace(-0.75, 1.)
       z_plot = np.zeros((len(x_plot), len(y_plot)))

       for i in range(len(x_plot)):
           for j in range(len(y_plot)):
              z_plot[i, j] = map_features(x_plot[i], y_plot[j]).dot(theta)

       z_plot = z_plot.T

       ax.scatter(x[posi, 1], x[posi, 2], marker='+', label='Microchip test 1')
       ax.scatter(x[neg, 1], x[neg, 2], c='r' , marker='x', label='Microchip test 2')
       contour = ax.contour(x_plot, y_plot, z_plot, levels=0, colors='g')
       contour.collections[1].set_label('Decision Boundary')
       ax.legend()

    plt.show()

def map_features(x1, x2, degree=6):
    """
        maps features to polynomials
    """
    # this if statement is neccesary when passing numbers only 
    # the type 'numpy.float64' has not the 'len' method
    if type(x1) != 'numpy.ndarray': 
        x1 = x1.flatten(); x2 = x2.flatten()

    out = np.ones(len(x1)).reshape(-1, 1)

    for i in range(1, degree + 1):
        for j in range(i + 1):
            this_power = np.power(x1, i - j) * np.power(x2, j)
            this_power = this_power.reshape(-1, 1)
            out = np.concatenate((out, this_power), axis=1)

    return out

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

