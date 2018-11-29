import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

from utils import display_data
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

display_data(X[rand_indxs, :])
