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

display_data(X)
