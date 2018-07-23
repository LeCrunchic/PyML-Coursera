import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_bfgs

from utils import costFunction

# Load Data
data = np.genfromtxt('data/ex2/ex2data1.txt', delimiter=',')

X = data[:, 0:2]
Y = data[:, [2]]

# ============== Part 1 - Plot Data ============= #

print('Plotting Data')
print('Positive examples are plotted with a plus sign')
print('Negative examples are plotted with an x')

posi = np.where(Y == 1)[0]
neg = np.where(Y == 0)[0]

print('Remember to close the plot. Otherwise, the process does not continue')

plt.scatter(X[posi, 0], X[posi, 1], marker='+')
plt.scatter(X[neg, 0], X[neg, 1], c='red' , marker='x')
plt.xlabel('Score: First exam')
plt.ylabel('Score: Second exam')
plt.legend(['Admitted', 'Not admitted'])
plt.show()

# ============= Part 2: Compute cost and gradient ============== #

print('Calculating cost and gradient...')

m, n = X.shape
X = np.concatenate((np.ones((m, 1)), X), axis=1)
initial_theta = np.zeros((n + 1, 1))
cost, grad = costFunction(initial_theta, X, Y)

print(f'Cost with initial parameters (all zeros): {cost}')
print(f'Gradient with initial parameters:\n{grad}')

test_theta = np.array([[-24],[0.2],[0.2]])
cost, grad = costFunction(test_theta, X, Y)

print(f'Cost with test parameters:\n{test_theta}\nCost:{cost}')
print(f'Gradient with test parameters: \n{grad}')

input('Press enter to continue...')

# ================= Part 3: Optimizing ================== #

print('Optimization using the BFGS algorithm...')

#fmin_bfgs(costFunction())



