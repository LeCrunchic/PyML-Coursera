import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_bfgs

from utils import compute_cost, compute_gradient, plot_data, plot_bounded_data, sigmoid
# Load Data
data = np.genfromtxt('data/ex2/ex2data1.txt', delimiter=',')

X = data[:, :2]
Y = data[:, -1]

# ============== Part 1 - Plot Data ============= #

print('Plotting Data')
print('Positive examples are plotted with a plus sign')
print('Negative examples are plotted with an x')

print('Remember to close the plot. Otherwise, the process does not continue')
xlabel = 'Score: First Exam'
ylabel = 'Score: Second Exam'
legend = ['Admitted', 'Not admitted']

plot_data(X, Y, xlabel, ylabel, legend)

# ============= Part 2: Compute cost and gradient ============== #                                                         

print('Calculating cost and gradient...')

m, n = X.shape
X = np.concatenate((np.ones((m, 1)), X), axis=1)
initial_theta = np.zeros((n + 1, 1))
cost = compute_cost(initial_theta, X, Y)
grad = compute_gradient(initial_theta, X, Y)

print(f'Cost with initial parameters (all zeros): {cost}')
print(f'Gradients with initial parameters:\n{grad}')

test_theta = np.array([[-24],[0.2],[0.2]])
cost = compute_cost(test_theta, X, Y)
grad = compute_gradient(test_theta, X, Y)

print(f'Cost with test parameters:\n{test_theta}\nCost:{cost}')
print(f'Gradients with test parameters: \n{grad}')

input('Press enter to continue...')

# ================= Part 3: Optimizing ================== #

print('Optimization using the BFGS algorithm...')
from functools import partial

f = partial(compute_cost, x=X, y=Y)
fprime = partial(compute_gradient, x=X, y=Y)

optimal_theta = fmin_bfgs(f, initial_theta, fprime, maxiter=400)
print("theta after optimization:", optimal_theta)

# Plotting w/decision boundary
plot_bounded_data(X, Y, optimal_theta)

# =============== Prediction ================== #
prob = sigmoid( np.array([1, 45, 85]).dot(optimal_theta) )

print(f"for an student with a first score of 45 and a second of 86 we predict a probability of admision of {prob}")

pred = sigmoid(X.dot(optimal_theta)) >= 0.5

accuracy = np.mean((pred == Y) * 100)

print(f'Train accuracy: {accuracy}')
