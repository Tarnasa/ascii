"""
Put it all together now,
"""

from multiblur import score_rect, compare_scores, generate_blurred_images
from glob import glob
from collections import namedtuple
from skimage import io
from scipy.misc import imresize
import numpy as np

from sklearn import svm
import random

Alphabet = namedtuple('Alphabet', ('scores', 'cw', 'ch'))

def fit(w, h, subw, subh):
    """
    Calculates the best width and height to scale the original rectangle
    to which minimizes change in width + height.  I think.
    """
    new_w = subw * int(w / float(subw) + 0.5)
    new_h = subh * int(h / float(subh) + 0.5)
    return new_w, new_h


def convert(image, alphabet, blur_levels=3):
    clf = svm_train(blur_levels)
    ch, cw = alphabet.ch, alphabet.cw
    h, w = image.shape
    w, h = fit(w, h, cw, ch)
    image = imresize(image*255, (h, w), interp='bilinear')
    image = image.astype(np.float32) / 255.
    blur_factor = min(cw/8, ch/8)/2
    blur_factor = 0
    images = generate_blurred_images(image, blur_factor, blur_levels)
    art = []
    for yi, y in enumerate(range(0, h, ch)):
        line = []
        for xi, x in enumerate(range(0, w, cw)):
            #print(xi, yi, x, y)
            scene = score_rect(images, x, y, cw, ch, cw/8, ch/8, blur_levels)
            #filename = find_best_char(scene, alphabet)
            filename = random.choice(clf.predict(scene.features))
            #if xi == 22 and yi == 20:
            #    print('hey', filename, compare_scores(scene, alphabet.scores[filename]))
            #    print('space', filename, compare_scores(scene, alphabet.scores['out/32.png']))
            #    print('light', scene.norm)
            #    print('light', alphabet.scores['out/126.png'].norm)
            #if xi == 33 and yi == 20:
            #    print('light', scene.norm)

            line.append(chr(int(filename[4:-4])))
        art.append(line)
    return '\n'.join(''.join(line) for line in art)


def find_best_char(scene, alphabet):
    return min(alphabet.scores.items(),
            key=lambda pair: compare_scores(scene, pair[1]))[0]

def load_image(filename):
    #print('loading ...', filename)
    image = io.imread(filename)
    image = image.astype(np.float32)
    m = np.max(image)
    image /= m
    if len(image.shape) == 3:
        image = 1 - np.min(image, axis=2)
    else:
        image = 1 - image
    return image


def generate_alphabet(pattern, levels=3):
    scenes = dict()
    filenames = glob(pattern)
    for filename in filenames:
        image = load_image(filename)
        h, w = image.shape
        blur_factor = min(w/8, h/8)/2
        blur_factor = 0
        images = generate_blurred_images(image, blur_factor, levels)

        if filename == '!out/126.png':
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.imshow(images[2], cmap='gray', interpolation='nearest')
            plt.show()

        scene = score_rect(images, 0, 0, w, h, w/8, h/8, levels)
        scenes[filename] = scene
    return Alphabet(scenes, w, h)

def svm_train(blur_levels=3):
    alphabet = generate_alphabet('out/*.png', blur_levels)
    X = []
    y = []
    for i in range(32, 126):
        img = load_image('out/{}.png'.format(i))
        h, w = img.shape
        images = generate_blurred_images(img, min(w/8, h/8)/2, blur_levels)
        scene = score_rect(images, 0, 0, w, h, w/8, h/8, blur_levels)
        #print(scene[0][1])

        # Use the blur levels as the training feature
        X.append(sum(scene[0]))
        y.append('out/{}.png'.format(i))

    clf = svm.SVC(gamma=0.001, C=100)

    X = np.array(X)
    y = np.array(y)
    #print(X)
    clf.fit(X,y)

    return clf

if __name__ == '__main__':
    alphabet = generate_alphabet('out/*.png')
    image = load_image('test.png')
    art = convert(image, alphabet, blur_levels=3)
    print(art)

