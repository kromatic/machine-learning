#!/usr/bin/env python3

import numpy as np
import sys

phi = np.vectorize(lambda x: 1/(1 + np.e**(-x)), otypes=[np.float])

def apply_nnet(dig_pixels, ws):
    res = [np.append(train_xs[i], 1)]
    for d in range(1, ws.shape[0]):
        res.append(phi(ws[d-1].dot(res[-1])))
    return res

def update_weights(ws, res, y, eta):
    deltas = [(res[-1] - y)*res[-1]*(1 - res[-1])]
    for d in range(len(res)-1, -1, -1):
        row = res[d]*(1 - res[d])*ws[d].T.dot(deltas[-1])
        deltas.append(row)
    for d in range(len(ws)):
        for t in range(ws[d].shape[0]):
            ws[d][t] = ws[d][t] - eta*deltas[-d-2][t]*res[d]

if __name__ == "__main__":
    xs_file, ys_file, test_file = sys.argv[1:]
    train_xs = np.loadtxt(xs_file, delimiter=",")
    train_ys = np.loadtxt(ys_file, dtype=np.int)
    test_xs = np.loadtxt(xs_file, delimiter=",")
    print(train_xs)
    print(train_ys)
    print(test_xs)
