#!/usr/bin/env python3

# Leo Gladkov

import sys
import numpy as np
import time
from glob import glob
from PIL import Image, ImageDraw, ImageColor

def construct_cascade(pos_indices, neg_indices, fpb=0.01, adb_fpb=0.3):
    total_neg = len(neg_indices)
    cascade = []
    cnt = 1
    while True:
        t0 = time.time()
        classifier, neg_indices, fpr = adaboost(pos_indices, neg_indices, adb_fpb)
        cascade.append(classifier)
        t1 = time.time()
        print("added classifier {} in {:.1f} minutes:".format(cnt, (t1-t0)/60))
        print("\tfpr={:.2f}".format(fpr))
        print("\tnumber of negative examples left: {}".format(len(neg_indices)))
        cnt += 1
        if len(neg_indices)/total_neg <= fpb:
            return cascade

def detect_faces(img_file, cascade, win, stepm=0.1, tol=0.2):
    step = int(stepm*win)
    with Image.open(img_file) as img:
        iimg = get_iimage(np.reshape(img.getdata(), (img.height, img.width)))
        img_color = img.convert("RGB")
        draw = ImageDraw.Draw(img_color)
        face_corners = set()
        for i in range(0, img.height-win+1, step):
            for j in range(0, img.width-win+1, step):
                if win_overlap_any((i, j), face_corners, dim, tol):
                    continue
                if apply_cascade(cascade, iimg[i:i+win, j:j+win]):
                    face_corners.add((i, j))
                    draw.rectangle((i, j, i+win, j+win), outline=(255, 0, 0))
        img_color.save("faces__" + img_file)
        img_color.show()

def win_overlap_any(c, corners, dim, tol):
    return any(win_overlap(c, c1, dim, tol) for c1 in corners)

def win_overlap(c1, c2, dim, tol):
    i, k = c1[0], c2[0]+dim
    j, l = c1[1], c2[1]+dim
    if c2[0] > c1[0]:
        i, k = c2[0], c1[0]+dim
    if c2[1] > c1[1]:
        j, l = c2[1], c1[1]+dim
    if k-i <= 0 or l-j <= 0:
        return False
    return (k-i)*(l-j)/(dim*dim) > tol

def apply_cascade(cascade, iimg_win):
    return all(apply_classifier(classifier, 0, iimg_win)
               for classifier in cascade)

def adaboost(pos_indices, neg_indices, fpb):
    w = np.zeros(iimages.shape[0])
    for i in pos_indices:
        w[i] = 1/(2*len(pos_indices))
    for i in neg_indices:
        w[i] = 1/(2*len(neg_indices))
    training_indices = pos_indices.union(neg_indices)
    classifier = []
    while True:
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
                                                  big_theta, iimages[i]))
        fpr = len(false_positives)/len(neg_indices)
        if fpr <= fpb:
            return classifier, false_positives, fpr
        # update data set weights
        for i in training_indices:
            if apply_weak_learner(h, iimages[i]) == ys[i]:
                w[i] *= err/(1-err)
        # normalize weights
        total = w.sum()
        for i in training_indices:
            w[i] /= total

def best_weak_learner(training_indices, w):
    res = min((optimize_weak_learner(j, training_indices, w)
                for j in range(feature_tbl.shape[0])), key=lambda t: t[1])
    return res

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
    learner = p, j, theta
    return learner, min_err

def apply_classifier(classifier, big_theta, iimg_win, sign=True):
    s = 0
    for alpha, h in classifier:
        s += alpha*apply_weak_learner(h, iimg_win)
    diff = s-big_theta
    if not sign:
        return diff
    return diff >= 0

def apply_weak_learner(learner, iimg_win):
    p, j, theta = learner
    return 1 if p*(compute_feature(j, iimg_win) - theta) >= 0 else -1

def compute_feature(j, iimg_win):
    rect1, rect2 = feature_tbl[j, 0], feature_tbl[j, 1]
    return rect_sum(rect1, iimg_win) - rect_sum(rect2, iimg_win)

def rect_sum(rect, iimg_win):
    i, j, k, l = rect
    a = iimg_win[i-1, j-1] if i > 0 and j > 0 else 0
    b = iimg_win[i-1, l] if i > 0 else 0
    c = iimg_win[k, j-1] if j > 0 else 0
    d = iimg_win[k, l]
    return a+d-b-c

def get_training_data(faces_dir, backgrounds_dir):
    faces, backgrounds = glob(faces_dir + "/*"), glob(backgrounds_dir + "/*")
    with Image.open(faces[0]) as img:
        dim = img.width
    iimages = np.empty((len(faces) + len(backgrounds), dim, dim), dtype=np.int)
    ys = np.ones(len(faces) + len(backgrounds), dtype=np.int)
    ys[len(faces):].fill(-1)
    pos_indices = set(range(len(faces)))
    neg_indices = set(range(len(faces), ys.shape[0]))
    read_training_images(faces, dim, iimages)
    read_training_images(backgrounds, dim, iimages[len(faces):])
    return iimages, ys, pos_indices, neg_indices, dim

def read_training_images(imgs, dim, res):
    size = (dim, dim)
    for i in range(len(imgs)):
        with Image.open(imgs[i]) as img:
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

def compute_feature_tbl(dim, step=4):
    res = []
    for h in range(1, dim+1, step):
        for w in range(1, dim+1, step):
            for i in range(0, dim-h+1, step):
                for j in range(0, dim-w+1, step):
                    if i+2*h <= dim:
                        res.append(((i, j, i+h-1, j+w-1),
                                    (i+h, j, i+2*h-1, j+w-1)))
                    if j+2*w <= dim:
                        res.append(((i, j, i+h-1, j+w-1),
                                    (i, j+w, i+h-1, j+2*w-1)))
    return np.array(res)

if __name__ == "__main__":
    faces_dir, backgrounds_dir, test_img = sys.argv[1:]
    data = get_training_data(faces_dir, backgrounds_dir)
    iimages, ys, pos_indices, neg_indices, dim = data
    feature_tbl = compute_feature_tbl(dim)
    cascade = construct_cascade(pos_indices, neg_indices)
    detect_faces(test_img, cascade, dim)
