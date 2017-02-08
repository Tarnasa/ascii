from sklearn import svm
from multiblur import score_rect, compare_scores, generate_blurred_images
from utility import generate_alphabet, load_image
import os
import numpy as np

def svm_single(alphabet, blur_levels=3):
    X = []
    y = []
    for i in range(32, 127):
        img = load_image('out/{}.png'.format(i))
        h, w = img.shape
        images = generate_blurred_images(img, min(w/8, h/8)/2, blur_levels)
        scene = score_rect(images, 0, 0, w, h, w/8, h/8, blur_levels)

        # Use the blurred image vectors to train
        fv = []
        for j in range(blur_levels):
            fv.extend(scene[0][j])

        X.append(fv)
        y.append(i)

    clf = svm.SVC()

    X = np.array(X)
    y = np.array(y)
    clf.fit(X,y)

    return clf

def svm_multisample(blur_levels=3):
    #print("Starting training...")
    directories = os.listdir('training/')
    ydim = blur_levels * 100 + 1

    clf = svm.SVC()
    data = np.loadtxt("training.dat")

    X = [d[1:] for d in data]
    y = [d[0] for d in data]

    print("{} training samples".format(len(y)))

    clf.fit(X,y)
    #print("Training complete")
    return clf