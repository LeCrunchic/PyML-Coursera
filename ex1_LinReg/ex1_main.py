import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from warmUp import identity_warmup
from cost import computeCost
from gradientDescent import leGradientDescent

def main():
    #============== Part 1 Warm Up ==============#
    
    print('Running Warm Up Exercise')
    idMatrix = identity_warmup()
    print('5x5 Identity Matrix:\n', idMatrix)
    input('Press enter to continue...')
    
    #============== Part 2 Plot Data =============#
    
    data1 = np.genfromtxt('data/ex1/ex1data1.txt', delimiter=',')
    X = data1[:,[0]]
    Y = data1[:,[1]]
    print('Remember to close the figure. Otherwise, the process will not continue.')
    plt.plot(X, Y, marker='x', color='red', linewidth=0)
    
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
    print(f'With parameters:\n {theta}.\nThe cost is: {J}')
    input('Press enter to continue...')

    print('Running gradient descent...\n')
    theta = leGradientDescent(X, Y, alpha, iterations, theta)
    print(f'Parameters after 1500 iterations: \n{theta}')

    print(f'Cost with trained parameters: {computeCost(X, Y, theta)}')
    print('Plotting linear fit...')
    print('Remember to close the figure. Otherwise, the program will not continue.')
    plt.scatter(X[:,[1]], Y, marker='x', color='red')
    plt.plot(X[:,[1]], X.dot(theta))
    plt.show()
     
    # ============ Visualizing J ============ #

    # grid over we will plot the cost
    theta0_values = np.linspace(-10, 10, 100)
    theta1_values = np.linspace(-1, 4, 100)

    J_values = np.zeros((len(theta0_values), len(theta1_values)))

    for i, iv in enumerate(theta0_values):
        for j, jv in enumerate(theta1_values):
            this_theta = np.array([[iv], [jv]])
            J_values[i, j] = computeCost(X, Y, this_theta) 
    
    print('Remember to close the figure. Otherwise, the process will not continue.')
    fig = plt.figure()
    ax = fig.add_subplot('111', projection='3d')
    ax.plot_surface(theta0_values, theta1_values, J_values)
    ax.text('Theta 0', 'Theta 1', 'Cost', 'CULO')
    plt.show()
    

if __name__ == "__main__":
    main()
