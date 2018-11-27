import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_bfgs

from utils import compute_cost, compute_gradient, plot_data, plot_bounded_data, map_features, sigmoid
# Regularized logistic regression 

data = np.genfromtxt('./data/ex2/ex2data2.txt', delimiter=',')

# ============== load and plot Data =============== #
X = data[:, :2]; Y = data[:, -1]

xlabel = 'Microchip test 1'
ylabel = 'Microchip test 2'
legend = ['y = 1', 'y = 0']

plot_data(X, Y, xlabel, ylabel, legend)

# =============  Map data to Polynomial features ============== #

# this maps the features to a polynomial of degree 6
X = map_features(X[:, 0], X[:, 1])

# ============= Run logistic regression ================ #
initial_theta = np.zeros(X.shape[1])

# Regularization Parameter: This dictates how much the cost function is penalized
initial_lambda = 1

print('Computing cost with initial theta...')
cost = compute_cost(initial_theta, X, Y, regularized=True, lambda_=initial_lambda)
grad = compute_gradient(initial_theta, X, Y, regularized=True, lambda_=initial_lambda)

print(f'Cost with initial theta: {cost}')
print(f'First 5 gradients with initial theta\n{grad[:5].reshape(-1, 1)}')

test_theta = np.ones(X.shape[1])
test_lambda = 10

print('Computing cost with test theta...')
cost = compute_cost(test_theta, X, Y, regularized=True, lambda_=test_lambda)
grad = compute_gradient(test_theta, X, Y, regularized=True, lambda_=test_lambda)

print(f'Cost with test theta: {cost}')
print(f'First 5 gradients with test theta\n{grad[:5].reshape(-1, 1)}')


# now actually doing the regresion
from functools import partial

f = partial(compute_cost, x=X, y=Y, regularized=True, lambda_=initial_lambda)
fprime = partial(compute_gradient, x=X, y=Y, regularized=True, lambda_=initial_lambda)

optimal_theta = fmin_bfgs(f, initial_theta, fprime, maxiter=400)

pred = sigmoid(X.dot(optimal_theta)) >= 0.5

print(f'Training accuracy: {np.mean((Y == pred) * 100)}')
# ============== Plot decision boundary =============== #

plot_bounded_data(X, Y, optimal_theta)












