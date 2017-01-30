#!/usr/bin/env python3

# Leo Gladkov

import sys
import numpy as np
import matplotlib.pyplot as plt

def isomap(data, p):
    """Reduce data to p dimensions using Isomap algorithm."""
    knng = knn_graph(data[:, :-1])
    shortest_distances = shortest_path_distances(knng)
    gram = np.empty(shortest_distances.shape)
    for i in range(gram.shape[0]):
        for j in range(gram.shape[0]):
            gram[i, j] = shortest_distances[i].dot(shortest_distances[j])
    eigenvals, eigenvectors = np.linalg.eigh(gram)
    eigenvals[:-p] = 0
    sqrt_big_lambda = np.diag(np.sqrt(eigenvals))
    res = eigenvectors.dot(sqrt_big_lambda[:, -p:])
    return np.concatenate((res, data[:, -1:]), axis=1)

def knn_graph(data, k=10):
    """Returns matrix of edge weights representing k-NN graph of data."""
    res = np.empty((data.shape[0], )*2)
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            res[i, j] = euclid_distance(data[i], data[j])
        cutoff = sorted(res[i])[k-1]
        for j in range(res.shape[1]):
            if res[i, j] > cutoff:
                res[i, j] = np.inf
    return res

def shortest_path_distances(a):
    """Compute all shortest path distances given matrix a representing graph."""
    res = np.copy(a)
    for k in range(res.shape[0]):
        for i in range(res.shape[0]):
            for j in range(res.shape[0]):
                res[i, j] = min(res[i, j], res[i, k] + res[k, j])
    return res

def euclid_distance(u, v):
    """Compute euclidian distance between vectors u and v."""
    return np.linalg.norm(u-v)

if __name__ == "__main__":
    filename, p = sys.argv[1], int(sys.argv[2])
    data = []
    with open(filename, "r") as data_file:
        for line in data_file:
            data.append([float(x) for x in line.split()])
    data = np.array(data)
    reduced_data = isomap(data, p)
    if p == 2:
        plt.title("Isomap Dimensionality Reduction")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1],
                    c=reduced_data[:, 2])
        plt.show()
