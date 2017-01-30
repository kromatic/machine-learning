#!/usr/bin/env python3

# Leo Gladkov

import sys
import numpy as np
import matplotlib.pyplot as plt

def pca(data, k):
    """Returns data written in terms of first k principal components."""
    assert k < data.shape[1]
    cov_matrix = np.cov(data[:, :-1], rowvar=False, bias=True)
    _, eigenvectors = np.linalg.eigh(cov_matrix)
    reduction_transformation = np.linalg.inv(eigenvectors)[-k:]
    reduced_data = np.empty((data.shape[0], k+1))
    for i in range(reduced_data.shape[0]):
        reduced_data[i, :-1] = reduction_transformation.dot(data[i, :-1])
    reduced_data[:, -1] = data[:, -1]
    return reduced_data

if __name__ == "__main__":
    filename = sys.argv[1]
    data = []
    with open(filename, "r") as data_file:
        for line in data_file:
            data.append([float(x) for x in line.split()])
    data = np.array(data)
    reduced_data = pca(data, 2)
    plt.title("PCA Dimensionality Reduction")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1],
                c=reduced_data[:, 2])
    plt.show()
