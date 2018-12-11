import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

from utils import display_data, compute_cost, compute_gradient, OneVsAll 
# Defining neccesary parameters
input_layer_size = 400 # 20x20 images
num_labels = 10

# ========== Loading and Visualizing data ========== #

print('Loading data...')

data = loadmat('data/ex3/ex3data1.mat')

X = data['X']
Y = data['y']

m = X.shape[0]
idx_array = np.arange(m)
rand_indxs = np.random.choice(idx_array, size=100, replace=False)

print('Plotting example digits...')
display_data(X[rand_indxs, :])

input('Press enter to continue...')

# ========== Test Logistic Regression ============ #

theta_t = np.array([-2, -1, 1, 2])
ones = np.ones((5, 1))

X_t = np.concatenate([ np.ones((5,1)) , np.arange(1, 16).reshape(5, 3, order='F') / 10], axis=1)
y_t = np.array([1, 0, 1, 0, 1]) >= 0.5
lambda_t = 3

cost = compute_cost(theta_t, X_t, y_t, regularized=True, lambda_=lambda_t)
grad = compute_gradient(theta_t, X_t, y_t, regularized=True,lambda_=lambda_t)

print(f'cost with test parameters: {cost}')
print(f'gradients with test parameters:\n{grad}')

input('Press enter to continue...')

# =============== Train One vs All =============== #

lambda_ = 0.1
m = OneVsAll(X, Y, num_labels, lambda_)
m.fit()

# =============== Predict One vs All =============== #

preds = m.predict()
preds += 1 # python starts counting from 0, the labels starts from 1

print(f'Training set accuracy:', np.mean(Y.flatten() == preds) * 100)
