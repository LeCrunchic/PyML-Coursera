import numpy as np
import matplotlib.pyplot as plt

from featureNormalization import normalize
from gradientDescent import leGradientDescent
from normalEqn import normal_equation


#============== Part 1 Feature Normalization ==============#

# The first column of X represents the area (ft^2) of the house and the
# second column represents the number of rooms 

data = np.genfromtxt('data/ex1/ex1data2.txt', delimiter=',')
X = data[:, 0:2]
Y = data[:, [2]]


print('Normalizing features...')
mu, sigma, X_norm = normalize(X) 

X_norm = np.concatenate((np.ones((X.shape[0], 1)), X_norm), axis=1)

# ============== Part 2 Gradient Descent =================#
 
alpha = 0.0000003
iterations = 400

# Initial parameters
parameters = np.array([[80000], [135], [-8000]])

# Gradient Descent
parameters, J_history = leGradientDescent(X_norm, Y, alpha, iterations, parameters, multi=True)

# Plot of Cost
print('Close the plot for the process to continue...')
plt.plot(np.arange(len(J_history)), J_history)
plt.xlabel('Number of Iterations')
plt.ylabel('Cost')
plt.show()

print('Parameters computed by gradient descent after 400 iterations:')
print(parameters)

print('Price of a house with an area of 1650ft^2 and 3 rooms:')
print(np.array([1, 1650, 3]).dot(parameters))

#============== Part 3 - Normal Equations ==============#

# Add intercept to non normalized X
X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

parameters = normal_equation(X, Y)

print('Parameters learned with normal equation:')
print(parameters)

print('Price of a house with an area of 1650ft^2 and 3 rooms:')
print(np.array([1, 1650, 3]).dot(parameters))





