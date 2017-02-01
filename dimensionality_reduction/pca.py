#!/usr/bin/env python3

# Leo Gladkov

"""
Apply PCA to dimensionally reduce data.

Default reduction is to 2 dimensions, but PCA is implemented for the general
case of reducing to m dimensions.

Usage: ./pca.py datafile_name
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

def pca(data, m):
    """Returns data written in terms of first m principal components."""
    cov_matrix = np.cov(data[:, :-1], rowvar=False, bias=True)
    _, eigenvectors = np.linalg.eigh(cov_matrix)
    reduction_transformation = np.linalg.inv(eigenvectors)[-m:]
    reduced_data = reduction_transformation.dot(data.T[:-1])
    reduced_data = np.concatenate((reduced_data.T, data[:, -1:]), axis=1)
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
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=reduced_data[:, 2])
    plt.show()
