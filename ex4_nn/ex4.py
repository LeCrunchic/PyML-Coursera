import numpy as np
from scipy.io import loadmat

from utils import display_data, nn_cost_function, sigmoid_gradient, rand_init_weights, check_nn_gradients

input_layer_size = 400
hidden_layer_size = 25
num_labels = 10

# ============= Visualizing data ============== #

print('Loading data...')

data = loadmat('data/ex4/ex4data1.mat')

X = data['X']
Y = data['y']

m = X.shape[0]
idx_array = np.arange(m)
rand_indxs = np.random.choice(idx_array, size=100, replace=False)

print('Plotting example digits...')
display_data(X[rand_indxs, :])

input('Press enter to continue...')

# ============= Loading NN parameters ============= #

print('Loading Neural Network weights...')

weights = loadmat('data/ex4/ex4weights.mat')

W1 = weights['Theta1'].flatten(order='F')[:, np.newaxis]
W2 = weights['Theta2'].flatten(order='F')[:, np.newaxis]

#import pdb; pdb.set_trace()

nn_params = np.vstack((W1, W2))
# ============= Compute cost ============== #

print('Feedforward using NN.')

# weight regularization parameter
lambda_ = 0

J, _ = nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, Y, lambda_)

print('Cost at loaded weights:', J)

input('Press enter to continue...')

# ============= Cost with regularization ============= #

lambda_ = 1

J, _ = nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, Y, lambda_)

print('Cost at loaded weights w/ regularization (lambda = 1):', J)

input('Press enter to continue...')

# ============== Sigmoid gradient ================ #

g = sigmoid_gradient(np.array([-1, -0.5, 0, 0.5, 1]))
print('Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]:', g)

# ============= Randomly initialize parameters ============== #

print('Initializing NN parameters...')

initial_theta1 = rand_init_weights(input_layer_size, hidden_layer_size)
initial_theta2 = rand_init_weights(hidden_layer_size, num_labels)

initial_nn_params = np.hstack(( initial_theta1.flatten(), initial_theta2.flatten() ))

# =============== Check NN gradients =============== #

check_nn_gradients()

# ============= Backpropagation with regularization ============= #

print('checking backpropagation w/ regularization')

check_nn_gradients(lambda_=3)

debug_J, _ = nn_cost_function(nn_params, input_layer_size, hidden_layer_size,
                              num_labels, X, Y, lambda_=3)

print(f'cost with lambda_ = 3, this value should be about 0.576051: {debug_J} ')
