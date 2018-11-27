import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression(data_path, regularized):
    def __init__(self, data_path, regularized=False):
        self.data = np.genfromtxt(data_path)
        self.X = self.data[:, :2]
        self.Y = self.data[:, -1]
        self.pos = self.Y == 1
        self.neg = self.Y == 0

        if regularized:
            mean = np.mean(self.X, axis=1)
            std = np.std(self.X, axis=1, ddof=1) 
            self.X = (self.X - mean) / std
            
    def optimize(self, initial_theta,) 


    def plot_data(self, xlabel, ylabel, legend):
        """
            xlabel: string
            ylabel: string
            legend: list
        """
        plt.scatter(self.X[self.pos, 0], self.X[self.pos, 1])
        plt.scatter(self.X[self.neg, 0], self.X[self.neg, 1])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(legend)
        plt.show()

    def plot_bounded_data(self, xlabel, ylabel, legend):
        plt.scatter(self.X[self.pos, 0], self.X[self.pos, 1])
        plt.scatter(self.X[self.neg, 0], self.X[self.neg, 1])
        



