#!/usr/bin/env python3

import numpy as np
from collections import defaultdict
import random

def kmeans(data, k):
    """Partition data into k clusters using the kmeans algorithm."""
    # get lower and upper bounds in data for each dimension
    if not data or k <= 0:
        raise ValueError
    n = data.shape[0]
    d = data.shape[1]
    cluster_ids = range(k)
    mins = np.vstack((data.min(axis=0),) * k)
    multipliers = np.vstack((data.max(axis=0) - mins,) * k)
    # initialize cluster centers to random numbers rescricted by bounds
    centers = mins + multipliers * np.random.random((k, d))
    prev_assignments = None    # used to check for terminating condition
    assignments = np.array([-1 for _ in data])   # begin with one dummy cluster
    while tuple(assignments) != prev_assignments:
        prev_assignments = tuple(assignments)
        for i in range(n):
            assignments[i] = min(cluster_ids,
                              key=lambda j: distance(data[i], centers[j]))

        clusters = defaultdict(set)
        for i in range(n):
            clusters[assignments[i]].add(data[i])
        for j in cluster_ids:
            centers[j] =  tuple(sum(x[k] for x in clusters[j])/len(clusters[j]) for k in range(d))
    return assignments
