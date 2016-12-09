"""
Put it all together now,
"""

from multiblur import score_rect, compare_scores, generate_blurred_images
from glob import glob
from collections import namedtuple
from skimage import io
from scipy.misc import imresize
import numpy as np
import ocr_svm
import time

Alphabet = namedtuple('Alphabet', ('scores', 'cw', 'ch'))

def fit(w, h, subw, subh):
    """
    Calculates the best width and height to scale the original rectangle
    to which minimizes change in width + height.  I think.
    """
    new_w = subw * int(w / float(subw) + 0.5)
    new_h = subh * int(h / float(subh) + 0.5)
    return new_w, new_h

def convert(image, alphabet, svm=None, blur_levels=3):
    ch, cw = alphabet.ch, alphabet.cw
    h, w = image.shape
    w, h = fit(w, h, cw, ch)
    image = imresize(image*255, (h, w), interp='bilinear')
    image = image.astype(np.float32) / 255.
    blur_factor = min(cw/8, ch/8)/2
    images = generate_blurred_images(image, blur_factor, blur_levels)
    art = []
    for yi, y in enumerate(range(0, h, ch)):
        line = []
        for xi, x in enumerate(range(0, w, cw)):
            scene = score_rect(images, x, y, cw, ch, cw/8, ch/8, blur_levels)
            if not svm:
                char = int(find_best_char(scene, alphabet)[4:-4])
            else:
                fv = []
                for j in range(blur_levels):
                    fv.extend(scene.features[j])
                fv = np.array(fv).reshape(1, -1)

                char = int(svm.predict(fv)[0])

            line.append(chr(char))
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
        images = generate_blurred_images(image, blur_factor, levels)

        if filename == '!out/126.png':
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.imshow(images[2], cmap='gray', interpolation='nearest')
            plt.show()
        scene = score_rect(images, 0, 0, w, h, w/8, h/8, levels)
        scenes[filename] = scene
    return Alphabet(scenes, w, h)

if __name__ == '__main__':
    alphabet = generate_alphabet('out/*.png')
    image = load_image('test.png')

    print("Nearest Neighbor Classification:")
    start = time.time()
    art = convert(image, alphabet, None, 3)
    print("Run time: {}s".format(time.time() - start))
    print(art)

    print("SVM with target font as training sample:")
    start = time.time()
    svm = ocr_svm.svm_single(3)
    print("Training time: {}s".format(time.time() - start))
    conv = time.time()
    art = convert(image, alphabet, svm, 3)
    print("Conversion time: {}s".format(time.time() - conv))
    print(art)

    print("SVM with many fonts as training samples:")
    svm = ocr_svm.svm_multisample(3)
    print("Training time: {}s".format(time.time() - start))
    conv = time.time()
    art = convert(image, alphabet, svm, 3)
    print("Conversion time: {}s".format(time.time() - conv))
    print(art)

