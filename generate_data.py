#!/usr/bin/env python2
from text import *
from matplotlib import font_manager, ft2font
import fs, os
from skimage import io
from multiblur import score_rect,  generate_blurred_images
from glob import glob
from collections import namedtuple
from subprocess import Popen
import numpy as np

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

def render_fonts():
    fonts = font_manager.fontManager.ttflist

    for font in fonts:
        name = font.name.replace(" ", "-") + str("-2")
        if "Mono" in name:
            try:
                os.mkdir('training/' + name)
                for code in ASCII:
                    render_text_to_file(chr(code), 'training/' + name + '/' + str(code) + '.png', font=name[:-2], size=32)   
            except OSError as err:
                pass
            


def extract_features(blur_levels=3):
    directories = os.listdir('training/')

    X = np.array([[0.0,]*(100*blur_levels + 1) for i in range(len(directories) * len(ASCII))])

    for d in range(len(directories)):
        directory = directories[d]
        path = 'training/' + directory
        print(path + '/*.png')
        alphabet = generate_alphabet(path + '/*.png', blur_levels)
        
        for i in ASCII:
            try:
                img = load_image('training/{}/{}.png'.format(directory, i))
                h, w = img.shape
                images = generate_blurred_images(img, min(w/8, h/8)/2, blur_levels)
                scene = score_rect(images, 0, 0, w, h, w/8, h/8, blur_levels)

                # Use the blurred image vectors as training data
                fv = []
                for j in range(blur_levels):
                    fv.extend(scene[0][j])

                X[d*len(ASCII) + i - ASCII[0]] = np.array([i] + fv)
            except Exception as err:
                X[d*len(ASCII) + i - ASCII[0]] = np.array([i] + ([0.0,] * (100*blur_levels)))
                print(err)
    
    print(X)
    np.savetxt("training.dat", X)

    print("Done! Generated {} training samples".format(len(directories) * len(ASCII)))

if __name__ == "__main__":
   #render_fonts()
   extract_features()
