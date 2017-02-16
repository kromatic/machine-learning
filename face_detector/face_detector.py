#!/usr/bin/env python3

import sys
import numpy as np
from glob import glob
from PIL import Image, ImageDraw, ImageColor

def construct_cascade(fp_bound=0.01):
    pos_indices, neg_indices = set(), set()
    for i in range(iimages.shape[0]):
        if ys[i] == 1:
            pos_indices.add(i)
        else:
            neg_indices.add(i)
    total_neg = len(neg_indices)
    cascade = []
    while True:
        classifier, neg_indices = adaboost(pos_indices, neg_indices)
        if len(neg_indices)/total_neg <= fp_bound:
            return cascade

def detect_faces(img_file, cascade, win):
    with Images.open(img_file) as img:
        pixels = np.reshape(img.convert("L").getdata(), img.size)
        iimg = get_iimage(pixels)
        draw = ImageDraw.Draw(img)
        for i in range(img.width-win+1):
            for j in range(img.height-win+1):
                if apply_cascade(cascade, iimg[i:i+win, j:j+win]) == 1:
                    draw.rectangle((i, j, i+win, j+win), fill=(255, 0, 0))
        img.show()

def apply_cascade(cascade, iimg_win):
    for classifier in cascade:
        if apply_classifier(classifier, 0, iimg_win) != 1:
            return -1
    return 1

def adaboost(pos_indices, neg_indices, fp_bound=0.3):
    w = np.zeros(iimages.shape[0])
    for i in pos_indices:
        w[i] = 1/len(pos_indices)
    for i in neg_indices:
        w[i] = 1/len(neg_indices)
    training_indices = pos_indices.union(neg_indices)
    classifier = []
    big_theta = 0
    while True:
        # normalize weights
        total = w.sum()
        for i in training_indices:
            w[i] /= total
        # find best weak learner
        h, err = best_weak_learner(training_indices, w)
        # compute weight and add to classifier
        alpha = np.log((1-err)/err)
        classifier.append((alpha, h))
        # update classifer threshold
        big_theta = min((apply_classifier(classifier, 0, iimages[i], sign=False)
                         for i in pos_indices))
        # check false positive rate
        false_positives = set(i for i in neg_indices
                              if apply_classifier(classifier,
                                                  big_theta, iimages[i]) == 1)
        if len(false_positives)/len(neg_indices) <= 0.3:
            return classifier, false_positives
        # update data set weights
        for i in correct_indices:
            if apply_weak_learner(h, iimages[i]) == ys[i]:
                w[i] *= err/(1-err)

def best_weak_learner(training_indices, w):
    return min((optimize_weak_learner(j, training_indices, w)
                for j in range(feature_tbl.shape[0])), key=lambda t: t[1])

def optimize_weak_learner(j, training_indices, w):
    vals = np.empty(iimages.shape[0])
    for i in training_indices:
        vals[i] = compute_feature(j, iimages[i])
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
    return learner, err

def apply_classifier(classifier, big_theta, iimg_win, sign=True):
    s = 0
    for alpha, h in classifier:
        s += alpha*apply_weak_learner(h, iimg_win)
    diff = s-big_theta
    if not sign:
        return diff
    return 1 if diff >= 0 else -1

def apply_weak_learner(learner, iimg_win):
    p, j, theta = learner
    return 1 if p*(compute_feature(j, iimg_win) - theta) >= 0 else -1

def compute_feature(j, iimg_win):
    rect1, rect2 = feature_tbl[j, 0], feature_tbl[j, 1]
    return rect_sum(rect1, iimg_win) - rect_sum(rect2, iimg_win)

def rect_sum(rect, iimg_win):
    i, j, k, l = rect
    A = 0 if i == 0 or j == 0 else iimg_win[i-1, j-1]
    B = 0 if i == 0 else iimg_win[i-1, l]
    C = 0 if j == 0 else iimg_win[k, j-1]
    D = iimg_win[k, l]
    return A + D - B - C

def get_training_data(faces_dir, backgrounds_dir):
    faces, backgrounds = glob(faces_dir + "/*"), glob(backgrounds_dir + "/*")
    with Image.open(faces[0]) as img:
        dim = img.height
    iimages = np.empty((len(faces) + len(backgrounds), dim, dim), dtype=np.int)
    ys = np.ones(len(faces) + len(backgrounds), dtype=np.int)
    ys[len(faces):].fill(-1)
    read_images(faces, dim, iimages)
    read_images(backgrounds, dim, iimages[len(faces):])
    return iimages, ys, dim

def read_images(imgs, dim, res):
    size = (dim, dim)
    for img_file in imgs:
        with Image.open(img_file) as img:
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
            for i in range(0, dim-h+1, step):
                for j in range(0, dim-w+1, step):
                    if i+2*h <= dim:
                        res.append(((i, j, i+h-1, j+w-1),
                                    (i+h-1, j, i+2*h-1, j+w-1)))
                    if j+2*w <= dim:
                        res.append(((i, j, i+h-1, j+w-1),
                                    (i, j+w-1, i+h-1, j+2*w-1)))
    return np.array(res)

if __name__ == "__main__":
    faces_dir, backgrounds_dir, test_img = sys.argv[1:]
    iimages, ys, dim = get_training_data(faces_dir, backgrounds_dir)
    feature_tbl = compute_feature_tbl(dim)
    print(iimages)
    print(ys)
    print(feature_tbl)
    cascade = construct_cascade()
    detect_faces(test_img, cascade, dim)
