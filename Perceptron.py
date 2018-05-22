import numpy as np
# Perceptron
from sklearn.linear_model import Perceptron
# Get data from datasets
from sklearn.datasets import make_classification
# plt
import matplotlib.pyplot as plt

X, y = make_classification(n_features=2,
                           n_redundant=0,
                           n_informative=2,
                           n_clusters_per_class=1)
p = Perceptron(max_iter = 1000)
p.fit(X, y)
maxX = -1000
minX = 1000
for i in range(len(X)):
	x0Value = X[i][0]
	if (x0Value > maxX):
		maxX = x0Value
	if (x0Value < minX):
		minX = x0Value
	if (y[i] == 0):
		plt.plot(X[i][0], X[i][1], 'bo')
	else:
		plt.plot(X[i][0], X[i][1], 'go')

#draw line
intercept = p.intercept_
w0 = p.coef_[0][0]
w1 = p.coef_[0][1]
x0 = np.linspace(minX * 1.1, maxX * 1., 20)
x1 = ((-intercept) - w0 * x0)/w1
plt.plot(x0, x1)
plt.show()