import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from sklearn.neighbors import NearestNeighbors

# Training points
X = [[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]]
# Query points
points = [[1.5,1.5]]
# k => how many nearest
k = 3
# Method1, explicitly using KDTree
def kdTreeMethod(X, k, points):
	kd = KDTree(X, leaf_size=30, metric='euclidean')
	_, indices = kd.query(points, k)
	return indices

# Method2 implicitly using KDTree. the sklearn API has neighbour function
def neighboursMethod(X, k, points):
	nb = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X)
	_, indices = nb.kneighbors(points)
	return indices

for p in X:
	plt.plot(p[0], p[1], 'bo')

for i in neighboursMethod(X, k, points)[0]:
	p = X[i]
	plt.plot(p[0], p[1], 'go')

plt.plot(points[0][0], points[0][1], color = 'red', marker = 'x')
plt.show()