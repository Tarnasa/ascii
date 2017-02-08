from glob import glob
from skimage import io
import numpy as np
from multiblur import score_rect, generate_blurred_images
from collections import namedtuple
import os, time, sys
from math import ceil
import difflib

ASCII = range(32, 127)
Alphabet = namedtuple('Alphabet', ('scores', 'cw', 'ch'))

def generate_alphabet(pattern, levels=3):
    scenes = dict()
    filenames = glob(pattern)
    for filename in filenames:
        image = load_image(filename)
        h, w = image.shape
        blur_factor = min(w/8, h/8)/2
        images = generate_blurred_images(image, blur_factor, levels)

        if filename == '!out/126.png':
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.imshow(images[2], cmap='gray', interpolation='nearest')
            plt.show()
        scene = score_rect(images, 0, 0, w, h, w/8, h/8, levels)
        scenes[filename] = scene
    return Alphabet(scenes, w, h)

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