#!/usr/bin/env python3

import numpy as np

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
    centers = [mins + multipliers * np.random.random(d) for _ in range(k)]
    # initialize clusters
    prev_clusters = None
    clusters = (set(range(n)),) + tuple(set() for _ in range(k-1))
    while clusters != prev_clusters:
        prev_clusters = tuple(cluster.copy() for cluster in clusters)
        for j in range(k):
            for i in tuple(clusters[j]):
                new_j = min(range(k), key=lambda j: dist(data[i], centers[j]))
                if dist(data[i], centers[new_j]) < dist(data[i], centers[j]):
                    clusters[j].remove(i)
                    clusters[new_j].add(i)
        for j in range(k):
            centers[j] = sum(data[i] for i in clusters[j]) / len(clusters[j])
    return clusters, centers

def dist(a, b):
    return np.sqrt(sum((a-b)**2))

def square_distance_cost(data, clusters, centers):
    return sum(distance(data[i], centers[j])**2
               for i in clusters[j] for j in range(len(clusters)))

if __name__ == "__main__":
    data = []
    while True:
        try:
            data.append([float(x) for x in input().split()])
        except EOFError:
            break
    print(kmeans(np.array(data), 3))
