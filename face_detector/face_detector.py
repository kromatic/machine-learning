#!/usr/bin/env python3

import sys
import numpy as np
from glob import glob
from PIL import Image

def adaboost(pos_indices, neg_indices, fp_bound=0.3):
    w = np.zeros(iimages.shape[0])
    for i in pos_indices:
        w[i] = 1/num_pos
    for i in neg_indices:
        w[i] = 1/neg_indices
    training_indices = pos_indices.union(neg_indices)
    classifier = []
    big_theta = 0
    while True:
        # normalize weights
        total = w.sum()
        for i in training_indices:
            w[i] /= total
        # find best weak learner
        h, err, correct_indices = best_weak_learner(training_indices, w)
        # compute weight and add to classifier
        alpha = np.log((1-err)/err)
        classifier.append((alpha, h))
        # update classifer threshold
        big_theta = min((apply_classifier(classifier, 0, i, sign=False)
                         for i in pos_indices))
        # check false positive rate
        false_positives = set(i for i in neg_indices
                              if apply_classifier(classifier,
                                                  big_theta, i) == 1)
        if len(false_positives)/len(neg_indices) <= 0.3:
            return classifier, false_positives
        # update data set weights
        for i in training_indices.intersection(correct_indices):
            w[i] *= err/(1-err)

def best_weak_learner(training_indices, w):
    return min((optimize_weak_learner(j, training_indices, w)
                for j in range(feature_tbl.shape[0])), key=lambda t: t[1])

def optimize_weak_learner(j, training_indices, w):
    vals = np.empty(iimages.shape[0])
    for i in training_indices:
        vals[i] = compute_feature(j, i)
    permutation = np.array(sorted(training_indices, key=lambda i: vals[i]))
    # use permutation to find training index with minimum error
    err_plus = sum(w[i] for i in permutation if ys[i] == -1)
    err_minus = 1-err_plus
    min_err, p = (err_plus, 1) if err_plus < err_minus else (err_minus, -1)
    min_err_i = 0
    for i in permutation:
        delta = ys[i]*w[i]
        err_plus += delta
        err_minus -= delta
        cand_err, cand_p = ((err_plus, 1) if err_plus < err_minus
                            else (err_minus, -1))
        if cand_err < min_err:
            min_err, p = cand_err, cand_p
            min_err_i = i
    theta = vals[min_err_i]
    learner = (p, j, theta)
    err = min_err
    correct_indices = set(i for i in training_indices
                          if apply_weak_learner(learner, i) == ys[i])
    return learner, err, correct_indices

def apply_classifier(classifier, big_theta, i, sign=True):
    s = 0
    for alpha, h in classifier:
        s += alpha*apply_weak_learner(h, i)
    diff = s-big_theta
    if not sign:
        return diff
    return 1 if diff >= 0 else -1

def apply_weak_learner(learner, i):
    p, j, theta = learner
    return 1 if p*(compute_feature(j, i) - theta) >= 0 else -1

def compute_feature(j, i):
    rect1, rect2 = feature_tbl[j, :2], feature_tbl[j, 2:]
    return rect_sum(rect1, i) - rect_sum(rect2, i)

def rect_sum(rect, i):
    c1, c4 = map(tuple, rect)
    c2 = c1[0], c4[1]
    c3 = c4[0], c1[1]
    iimg = iimages[i]
    return iimg[c1] + iimg[c4] - iimg[c2] - iimg[c3]

def get_training_data(faces_dir, backgrounds_dir):
    faces, backgrounds = glob(faces_dir + "/*"), glob(backgrounds_dir + "/*")
    with Image.open(faces[0]) as img:
        dim = img.height
    iimages = np.empty((len(faces) + len(backgrounds), dim, dim), dtype=np.int)
    ys = np.ones(len(faces) + len(backgrounds), dtype=np.int)
    ys[len(faces):].fill(-1)
    read_images(faces, faces_dir, dim, iimages)
    read_images(backgrounds, backgrounds_dir, dim, iimages[len(faces):])
    return iimages, ys, dim

def read_images(imgs, imgs_dir, dim, res):
    size = (dim, dim)
    for i in range(len(imgs)):
        if not imgs[i].startswith("."):
            with Image.open(imgs_dir + "/" + imgs[i]) as img:
                pixels = np.reshape(img.convert("L").getdata(), size)
                res[i] = get_iimage(pixels)

def get_iimage(pixels):
    res = np.empty(pixels.shape, dtype=np.int)
    for i in range(pixels.shape[0]):
        row_sum = 0
        for j in range(pixels.shape[1]):
            row_sum += pixels[i, j]
            above = res[i-1, j] if i > 0 else 0
            res[i, j] = row_sum + above
    return res

def compute_feature_tbl(dim, step=3):
    res = []
    for h in range(1, dim+1, step):
        for w in range(1, dim+1, step):
            for i in range(dim-h+1, step):
                for j in range(dim-w+1, step):
                    if i-1 + 2*h < dim:
                        res.append(((i, j), (i+h, j+w), (i+h, j), (i+2*h, j+w)))
                    if j-1 + 2*w < dim:
                        res.append(((i, j), (i+h, j+w), (i, j+w), (i+h, j+2*w)))
    return np.array(res)

if __name__ == "__main__":
    faces_dir, backgrounds_dir = sys.argv[1:]
    iimages, ys, dim = get_training_data(faces_dir, backgrounds_dir)
    feature_tbl = compute_feature_tbl(dim)
    print(iimages)
    print(ys)
    print(feature_tbl)
