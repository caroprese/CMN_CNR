import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Conferences.SIGIR.CMN_our_interface.Pinterest.PinterestICCVReader import PinterestICCVReader


def get_popularity(matrix):
    return matrix.A.sum(axis=0)


def get_percentile(popularity_array, k):
    # print('Luciano > popularity_array:', popularity_array)
    sorted_popularity_array = np.sort(popularity_array)
    index = int(round(popularity_array.shape[0] * k / 100))
    # print('Luciano > len:', popularity_array.shape[0])
    # print('Luciano > index:', index)
    percentile = sorted_popularity_array[index]
    # print('Luciano > percentile:', percentile)
    return percentile


dataset = PinterestICCVReader()
URM_train = dataset.URM_train.copy()
popularity = get_popularity(URM_train)

df = pd.DataFrame(data=popularity, columns=['pop'])
grs = df.groupby('pop')

matrix = np.zeros((len(grs), 2))

i = 0

for gr in grs:
    matrix[i, 0] = gr[0]
    matrix[i, 1] = gr[1].count()
    i += 1

alpha, beta, scale, pi = 1, 40, 50, np.pi
percentile = get_percentile(popularity, 50)

f = 1 / (beta * np.sqrt(2 * pi))

gamma = np.tanh(alpha * popularity) + \
        scale * f * np.exp(-1 / (2 * (beta ** 2)) * (popularity - percentile)**2)
"""
plt.figure()
plt.plot(matrix[:, 0], matrix[:, 1])
plt.xlabel('Popularity')
plt.ylabel('Frequency')
plt.show()
"""

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Popularity')
ax1.set_ylabel('Frequency', color=color)
ax1.plot(matrix[:, 0], matrix[:, 1], '.', color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Gamma', color=color)  # we already handled the x-label with ax1
ax2.plot(popularity, gamma, '.', color=color)
ax2.tick_params(axis='y', labelcolor=color)

plt.xscale("log")
plt.show()
