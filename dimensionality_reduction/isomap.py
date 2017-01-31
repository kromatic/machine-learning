#!/usr/bin/env python3

# Leo Gladkov

import sys
import numpy as np
import matplotlib.pyplot as plt

def isomap(data, m):
    """Reduce data to m dimensions using Isomap algorithm."""
    # compute matrix of shortest distances
    knn_matrix = knn_graph(data[:, :-1])
    distances_squared = shortest_path_distances(knn_matrix)**2
    # compute gram matrix
    n = data.shape[0]
    p = np.identity(n) - np.ones((n, n))/n
    gram = -p.dot(distances_squared).dot(p)/2
    # use eigendecomposition to complete mds
    eigenvals, eigenvectors = np.linalg.eigh(gram)
    diagonal = np.concatenate(([0 for _ in range(n-m)],
                               np.sqrt(eigenvals[-m:])))
    sqrt_big_lambda = np.diag(diagonal)
    res = eigenvectors.dot(sqrt_big_lambda[:, -m:])
    res = np.concatenate((res, data[:, -1:]), axis=1)
    return res

def knn_graph(data, k=None):
    """Returns symmetric matrix of edge weights in k-NN graph of data."""
    if k is None:
        k = min(10, data.shape[0])
    res = np.empty((data.shape[0], )*2)
    for i in range(res.shape[0]):
        for j in range(i+1):
            res[i, j] = res[j, i] = euclid_distance(data[i], data[j])
    kth_neighbors = np.sort(res)[:, k-1]
    for i in range(res.shape[0]):
        for j in range(i+1):
            if res[i, j] > kth_neighbors[i] and res[i, j] > kth_neighbors[j]:
                res[i, j] = res[j, i] = np.inf
    return res

def shortest_path_distances(a):
    """Compute all shortest path distances given matrix a representing graph."""
    res = np.copy(a)
    for k in range(res.shape[0]):
        for i in range(res.shape[0]):
            for j in range(i+1):
                res[i, j] = res[j, i] = min(res[i, j], res[i, k] + res[k, j])
    return res

def euclid_distance(u, v):
    """Compute euclidian distance between vectors u and v."""
    return np.linalg.norm(u-v)

if __name__ == "__main__":
    filename = sys.argv[1]
    data = []
    with open(filename, "r") as data_file:
        for line in data_file:
            data.append([float(x) for x in line.split()])
    data = np.array(data)
    reduced_data = isomap(data, 2)
    plt.title("Isomap Dimensionality Reduction")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1],
                c=reduced_data[:, 2])
    plt.show()
