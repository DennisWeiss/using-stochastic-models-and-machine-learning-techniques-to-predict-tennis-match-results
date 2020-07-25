import math
import random
import matplotlib.pyplot as plt
import numpy as np


def f(x, y):
    mu = 0.7
    sigma = 0.04
    return -(1 / (math.sqrt(2 * math.pi) * sigma) * math.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + 0.6 * math.sin(10 * y))


def get_z(f, x, y):
    z = np.zeros((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            z[i, j] = f(x[i], y[j])
    return z


def grid_search_points():
    points = []
    for i in [0.2, 0.4, 0.6, 0.8]:
        for j in [0.2, 0.4, 0.6, 0.8]:
            points.append((i, j))
    return points


def random_search_points():
    points = []
    for i in range(16):
        points.append((random.random(), random.random()))
    return points


x = np.arange(0, 1, 0.01)
y = np.arange(0, 1, 0.01)
z = get_z(f, x, y)

points = grid_search_points()

plt.plot(list(map(lambda x: x[0], points)), list(map(lambda x: x[1], points)), 'o', color='black')
f_contour_plot = plt.contourf(x, y, z, levels=50, cmap='Blues_r')
axes = plt.gca()
axes.set_xlim([0, 1])
axes.set_ylim([0, 1])
axes.set_xlabel('hyperparameter 1')
axes.set_ylabel('hyperparameter 2')
plt.colorbar(f_contour_plot)
plt.show()
