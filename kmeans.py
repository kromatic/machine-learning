#!/usr/bin/env python3

# Leo Gladkov

import sys
import numpy as np
import matplotlib.pyplot as plt

def random_centers(data, k):
    """Initialize random cluster centers for the k-means algorithm."""
    # check for faulty input
    if not len(data) or k <= 0:
        raise ValueError
    d = len(data[0])
    # initialize cluster centers to random numbers restricted by bounds
    mins = data.min(axis=0)
    multipliers = data.max(axis=0) - mins
    centers = np.array([mins + multipliers * np.random.random(d)
                        for _ in range(k)])
    return centers

def kmeans(data, k, initializer=random_centers):
    """Partition data into k clusters using the k-means algorithm."""
    # set up
    n = len(data)
    centers = initializer(data, k)
    prev_clusters = None
    clusters = (set(range(n)),) + tuple(set() for _ in range(k-1))
    costs = []
    while clusters != prev_clusters:
        prev_clusters = tuple(cluster.copy() for cluster in clusters)
        for j in range(k):
            for i in tuple(clusters[j]):
                new_j = min(range(k), key=lambda j: distance(data[i], centers[j]))
                if distance(data[i], centers[new_j]) < distance(data[i], centers[j]):
                    clusters[j].remove(i)
                    clusters[new_j].add(i)
        for j in range(k):
            if len(clusters[j]):
                centers[j] = sum(data[i] for i in clusters[j]) / len(clusters[j])
        costs.append(square_distance_cost(data, clusters, centers))
    return clusters, centers, costs

def kmeans_pp(data, k):
    """Initialize cluster centers for k-means using k-means++."""
    n = len(data)
    centers = [data[np.random.choice(range(n))]]
    for i in range(1, k):
        weights = np.array([min(distance(x, centers[p]) for p in range(i))**2
                            for x in data])
        weights *= 1 / sum(weights)
        centers.append(data[np.random.choice(range(n), p=weights)])
    return np.array(centers)

def distance(a, b):
    """Euclidian distance between a and b."""
    return np.sqrt(sum((a-b)**2))

def square_distance_cost(data, clusters, centers):
    """J_avg^2 cost of cluster configuration."""
    return sum(distance(data[i], centers[j])**2
               for j in range(len(clusters)) for i in clusters[j])

if __name__ == "__main__":
    k, filename = int(sys.argv[1]), sys.argv[2]
    data = []
    with open(filename, 'r') as f:
        for line in f:
            data.append([float(x) for x in line.split()])
    data = np.array(data)
    if len(data[0]) == 2:
        clusters, centers, _ = kmeans(data, k)
        # first plot the clusters
        # I referred to a stackoverflow answer about plotting
        plt.title("toydata.txt Clusters")
        plt.xlabel("x")
        plt.ylabel("y")
        for cluster in clusters:
            plt.plot([data[i, 0] for i in cluster], [data[i, 1] for i in cluster], '.')
        plt.plot(centers[:,0], centers[:,1], 'xk', markersize=8, markeredgewidth=3)
        # then plot convergence of k-means
        plt.figure(2)
        plt.title(r"Convergence of vanilla $k$-means")
        plt.xlabel("# of iterations")
        plt.ylabel(r"$J_{avg^2}$")
        for _ in range(20):
            _, _, costs = kmeans(data, k)
            plt.plot(costs)
        # now plot convergence of k-means with k-means++
        plt.figure(3)
        plt.title(r"Convergence of $k$-means with $k$-means$++$")
        plt.xlabel("# of iterations")
        plt.ylabel(r"$J_{avg^2}$")
        for _ in range(20):
            _, _, costs = kmeans(data, k, initializer=kmeans_pp)
            plt.plot(costs)
        plt.show()
