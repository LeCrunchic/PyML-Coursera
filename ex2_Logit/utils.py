import numpy as np
import matplotlib.pyplot as plt

def compute_cost(theta, x, y):
    m = y.shape[0]
    J = -(1./m) * np.sum(y * np.log(sigmoid(x.dot(theta))) + (1. - y) * np.log(1. - sigmoid(x.dot(theta))) ) 
    return J

def compute_gradient(theta, x, y):
    m = y.shape[0] 
    grad = (1./m) * ((sigmoid(x.dot(theta)) - y).T.dot(x)).T 
    return grad

def plot_data(x, y):
    posi = y == 1
    neg = y == 0

    plt.scatter(x[posi, 0], x[posi, 1], marker='+')
    plt.scatter(x[neg, 0], x[neg, 1], c='red' , marker='x')
    plt.xlabel('Score: First exam')
    plt.ylabel('Score: Second exam')
    plt.legend(['Admitted', 'Not admitted'])
    plt.show()           

def plot_bounded_data(x, y, theta):

    posi = y == 1
    neg = y == 0
    
    plt.scatter(x[posi, 1], x[posi, 2], marker='+')
    plt.scatter(x[neg, 1], x[neg, 2], c='r' , marker='x')

    plot_x = np.array([min(x[:, 1]), max(x[:, 1])])
    plot_y =  (-1. / theta[1]) * (theta[2] * plot_x + theta[0])   
    plt.plot(plot_x, plot_y, linestyle='--', c='g')                                      
    
    plt.xlabel('Score: First exam')
    plt.ylabel('Score: Second exam')

    plt.legend(['Boundary','Admitted', 'Not admitted'])
    plt.show()           

def sigmoid(x):
    return 1. / (1. + np.exp(-x))
