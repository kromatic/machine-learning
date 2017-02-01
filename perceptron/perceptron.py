#!/usr/bin/env python3

# Leo Gladkov

"""
Implementations of binary and multiclass perceptron algorithms.

usage: ./perceptron.py train_xs_file train_ys_file test_xs_file [--multiclass]
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

def predict(train_xs, train_ys, test_xs, multiclass):
    """Use perceptron algorithm to predict classifcation of train_xs."""
    if not multiclass:
        w, mistakes = binary_perceptron(train_xs, train_ys)
        predict_ys = np.apply_along_axis(lambda x: 1 if w.dot(x) >= 0 else -1,
                                         1, test_xs)
    else:
        ws, mistakes = multiclass_perceptron(train_xs, train_ys)
        predict_ys = np.apply_along_axis(multiclassify, 1, test_xs, ws)
    return predict_ys, mistakes

def binary_perceptron(train_xs, train_ys, m=5):
    """Learn train data using online binary perceptron m times."""
    w = np.zeros(train_xs.shape[1])
    mistakes = [0]
    for _ in range(m):
        for i in range(train_xs.shape[0]):
            if train_ys[i]*w.dot(train_xs[i]) <= 0:
                mistakes.append(mistakes[-1]+1)
                w += train_ys[i]*train_xs[i]
            else:
                mistakes.append(mistakes[-1])
    mistakes = np.array(mistakes)
    return w, mistakes

def multiclass_perceptron(train_xs, train_ys, m=5):
    """Learn train data using online binary perceptron m times."""
    num_classes = len(set(train_ys))
    ws = np.zeros((num_classes, train_xs.shape[1]))
    mistakes = [0]
    for _ in range(m):
        for i in range(train_xs.shape[0]):
            guess = multiclassify(train_xs[i], ws)
            if guess != train_ys[i]:
                mistakes.append(mistakes[-1]+1)
                ws[guess] -= train_xs[i]/2
                ws[train_ys[i]] += train_xs[i]/2
            else:
                mistakes.append(mistakes[-1])
    mistakes = np.array(mistakes)
    return ws, mistakes

def multiclassify(x, ws):
    """Classify x according to maximum dot product with vector from ws."""
    return np.argmax(ws.dot(x))

if __name__ == "__main__":
    # parse all data
    all_data = [], [], []
    for i in range(3):
        this_data = all_data[i]
        with open(sys.argv[i+1]) as f:
            first_line_features = [float(x) for x in f.readline().split()]
            if len(first_line_features) > 1:
                this_data.append(first_line_features)
                for line in f:
                    this_data.append([float(x) for x in line.split()])
            else:
                this_data.append(int(first_line_features[0]))
                for line in f:
                    this_data.append(int(line))
    multiclass = False
    if len(sys.argv) > 4 and sys.argv[4] == "--multiclass":
        multiclass = True
    # convert everything into numpy arrays
    train_xs, train_ys, test_xs = map(np.array, all_data)
    predict_ys, mistakes = predict(train_xs, train_ys, test_xs, multiclass)
    with open(sys.argv[3].rstrip("digits") + "predictions", "w") as f:
        for y in predict_ys:
            f.write("{}\n".format(y))
    plt.title(multiclass*"Multiclass " + "Perceptron Training Mistakes")
    plt.xlabel("# of examples")
    plt.ylabel("# of mistakes")
    plt.plot(mistakes)
    plt.show()
