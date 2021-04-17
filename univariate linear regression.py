import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = '/home/bakr/Documents/data1.txt'
data = pd.read_csv(path, header = None, names = ['population', 'profit'])
data.insert(0, 'ones', 1)
print('Data header: \n' , data.head(10))
print('-------------------------------------')

data.plot(kind = 'scatter', x = 'population', y = 'profit')

x, y = data.iloc[:, 0:2], data.iloc[:, 2:3]
m = y.size

x = np.matrix(x.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0, 0]))

def computeCost(x, y, theta):
    j = np.power(((x * theta.T)- y), 2)
    return np.sum(j) / (2 * m)
    

def gradientDescent(x, y, theta, alpha, num_itres):
    j_history = np.zeros(num_itres)
    parametars = int(theta.ravel().shape[1])
    temp = np.matrix(np.zeros(theta.shape))
    
    for i in range(num_itres):
        cost = (x * theta.T)- y
        for n in range(parametars):
            term = np.multiply(cost, x[:, n])
            temp[0, n] = theta[0, n] - ((alpha / m) * np.sum(term))
        theta = temp
        j_history[i] = computeCost(x, y, theta)
    return theta, j_history
    
alpha = 0.01
iters = 1500
theta, j_history = gradientDescent(x, y, theta, alpha, iters)

#plotting the line that fit the data
plt.plot(x[:, 1], np.dot(x, theta.T), '-')
plt.legend(['prediction', 'Training data'])

#plotting Error curve
fig, ax = plt.subplots()
ax.plot(np.arange(iters), j_history, 'r')
ax.set_xlabel("Iterations")
ax.set_ylabel("cost")
ax.set_title("Error curve")
