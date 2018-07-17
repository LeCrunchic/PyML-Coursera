import numpy as np
import matplotlib.pyplot as plt

# Load Data
data = np.genfromtxt('data/ex2/ex2data1.txt', delimiter=',')

X = data[:, 0:2]
Y = data[:, [2]]

# ============== Part 1 - Plot Data ============= #

print('Positive examples are plotted with a plus sign')
print('Negative examples are plotted with an x')

posi = np.where(Y == 1)[0]
neg = np.where(Y == 0)[0]

plt.scatter(X[posi, 0], X[posi, 1], marker='+')
plt.scatter(X[neg, 0], X[neg, 1], c='red' , marker='x')
plt.xlabel('Score: First exam')
plt.ylabel('Score: Second exam')
plt.show()





