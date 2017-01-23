"""
Put it all together now,
"""

from multiblur import score_rect, compare_scores, generate_blurred_images
from scipy.misc import imresize
import numpy as np
import sys
import ocr_svm
import time
from utility import generate_alphabet, load_image


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

if __name__ == '__main__':
    alphabet = generate_alphabet('out/*.png', 1)
    
    if len(sys.argv) < 2:
        print("Usage: python convert.py <image file>")
        sys.exit()
    image = load_image(sys.argv[1])

    print("Nearest Neighbor Classification:")
    start = time.time()
    art = convert(image, alphabet, None, 1)
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

    print("SVM with characters rendered in multiple fonts as training samples:")
    svm = ocr_svm.svm_multisample(3)
    print("Training time: {}s".format(time.time() - start))
    conv = time.time()
    art = convert(image, alphabet, svm, 1)
    print("Conversion time: {}s".format(time.time() - conv))
    print(art)