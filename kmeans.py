#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt

def kmeans(data, k):
    """Partition data into k clusters using the k-means algorithm."""
    # check for faulty input
    if not data.shape[0] or k <= 0:
        raise ValueError
    # set up
    n = data.shape[0]
    d = data.shape[1]
    # initialize cluster centers to random numbers restricted by bounds
    mins = data.min(axis=0)
    multipliers = data.max(axis=0) - mins
    centers = np.array([mins + multipliers * np.random.random(d) for _ in range(k)])
    # initialize clusters
    prev_clusters = None
    clusters = (set(range(n)),) + tuple(set() for _ in range(k-1))
    costs = []
    while clusters != prev_clusters:
        prev_clusters = tuple(cluster.copy() for cluster in clusters)
        for j in range(k):
            for i in tuple(clusters[j]):
                new_j = min(range(k), key=lambda j: dist(data[i], centers[j]))
                if dist(data[i], centers[new_j]) < dist(data[i], centers[j]):
                    clusters[j].remove(i)
                    clusters[new_j].add(i)
        for j in range(k):
            centers[j] = (sum(data[i] for i in clusters[j]) / len(clusters[j])
                          if len(clusters[j]) else
                          mins + multipliers * np.random.random(d))
        costs.append(square_distance_cost(data, clusters, centers))
    return clusters, centers, costs

def dist(a, b):
    return np.sqrt(sum((a-b)**2))

def square_distance_cost(data, clusters, centers):
    return sum(dist(data[i], centers[j])**2
               for j in range(len(clusters)) for i in clusters[j])

if __name__ == "__main__":
    k, filename = int(sys.argv[1]), sys.argv[2]
    data = []
    with open(filename, 'r') as f:
        for line in f:
            data.append([float(x) for x in line.split()])
    data = np.array(data)
    clusters, centers, costs = kmeans(data, 3)
    if data.shape[1] == 2:
        plt.title("toydata.txt Clusters")
        plt.xlabel("x")
        plt.ylabel("y")
        for cluster in clusters:
            plt.plot([data[i, 0] for i in cluster], [data[i, 1] for i in cluster], '.')
        plt.plot(centers[:,0], centers[:,1], 'xk', markersize=8, markeredgewidth=3)

        plt.figure(2)
        plt.title(r"Vanilla $k$-means Convergence")
        plt.xlabel("# of iterations")
        plt.ylabel(r"$J_{avg^2}$")
        for _ in range(20):
            _, _, costs = kmeans(data, 3)
            plt.plot(costs)
        plt.show()
