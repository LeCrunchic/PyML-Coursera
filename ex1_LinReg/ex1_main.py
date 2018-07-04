import numpy as np
import matplotlib.pyplot as plt

from warmUp import identity_warmup
from plotData import leplot
from cost import computeCost

def main():
    #============== Part 1 Warm Up ==============#
    
    print('Running Warm Up Exercise')
    idMatrix = identity_warmup()
    print('5x5 Identity Matrix:\n', idMatrix)
    input('Press enter to continue...')
    
    #============== Part 2 Plot Data =============#
    
    data1 = np.genfromtxt('data/ex1/ex1data1.txt', delimiter=',')
    X = data1[:,[0]]
    Y = data1[:,1]
    fig, ax = plt.subplots(1, 1)
    leplot(ax, X, Y, {'marker':'x','color':'red', 'linewidth':0})
    plt.show()                                                       

    #============== Part 3 Cost and Gradient descent =============#
    m = X.shape[0] #number of training examples
    ones = np.ones((m,1))
    X = np.concatenate((ones, X), axis=1) # add column of ones to X
    theta = np.zeros((2,1)) # initialize parameters

    #hyperparameters
    iterations = 1500
    alpha = 0.01 #learning rate

    print('Running cost function...')
    J = computeCost(X, Y, theta)
    print(f'With parameters:\n {theta}. The cost is:\n {J}')
    input('Press enter to continue...')

if __name__ == "__main__":
    main()