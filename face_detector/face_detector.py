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
    classifier = []
    big_theta = 0
    while True:
        # normalize weights
        total = np.sum(w)
        for i in training_indices:
            w[i] /= total
        # find best weak learner
        learner, err, correct_indices = best_weak_learner(training_indices, w)
        # compute weight and add to classifier
        alpha = np.log((1-err)/err)
        strong_learner.append((alpha, learner))
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
    

def get_training_data(faces_dir, backgrounds_dir):
    faces, backgrounds = glob(faces_dir + "/*"), glob(backgrounds_dir + "/*")
    with Image.open(faces[0]) as img:
        dim = img.height
    iimages = np.empty((len(faces) + len(backgrounds), dim, dim))
    ys = np.ones(len(faces) + len(backgrounds))
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
    res = np.empty(pixels.shape)
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
