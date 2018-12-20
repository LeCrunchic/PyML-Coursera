import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

from utils import display_data, NeuralNetwork

input_layer_size = 400
hidden_layer_size = 25
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

# ============= Loading weights for the NN ============= #

weights = loadmat('data/ex3/ex3weights.mat')
w1 = weights['Theta1']
w2 = weights['Theta2']

# ============= Making Predictions ============== #
nn = NeuralNetwork(X, Y, w1, w2)

preds = nn.predict()
preds += 1

#import pdb; pdb.set_trace()

print(f'Prediction accuracy: {np.mean(preds == nn.y.flatten()) * 100}')

# ============= Testing neural network ============= # 
rp_idx_array = np.random.permutation(idx_array)

for i in range(m):
    print('Displaying example...')
    display_data(X[rp_idx_array[i], :][np.newaxis, :])

    pred = nn.predict(train=False, test_x=X[rp_idx_array[i], :][np.newaxis, :])
    pred += 1

    print(f'Neural network says it is a {pred[0]}')
    salir = input('Close the example. Press enter if you want to continue, press ´y´ and then enter to exit otherwise: ')

    if salir == 'y':
        break
