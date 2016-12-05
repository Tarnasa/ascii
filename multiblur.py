import numpy as np

from scipy.ndimage.filters import gaussian_filter
from collections import namedtuple

BlurScene = namedtuple('BlurScene', ('features', 'norm'))


def calculate_normalizer(scores):
    v = np.sum(scores) / sum(scores.shape)
    if v < 0:
        print('NO!', v, scores)
    return v

def generate_blurred_images(image, blur_factor, levels):
    images = []
    for blur_level in range(levels):
        images.append(gaussian_filter(image, blur_factor*blur_level, mode='reflect', cval=0))
    # TODO 'constant', or 'reflect'
    return images

def score_rect(images, x, y, w, h, xs, ys, levels):
    #features = np.zeros((levels, ((h-1)/ys) * ((w-1)/xs)))
    features = np.zeros((levels, ((h-1)/ys+1) * ((w-1)/xs+1)), dtype=np.float32)
    blur_factor = min(xs, ys) / 2
    for blur_level in range(levels):
        blurred = images[blur_level]
        features[blur_level] = blurred[y:y+h:ys, x:x+w:xs].ravel()
    return BlurScene(features, calculate_normalizer(features))

def compare_scores(scene1, scene2):
    return np.sum(np.linalg.norm(scene1.features - scene2.features, ord=1, axis=1)) / ((scene1.norm + scene2.norm) or 0.01)

def draw_blur_levels():
    import matplotlib.pyplot as plt
    from skimage import io

    image = io.imread('out/66.png')  # 36 for $, 79 for O

    fig, axes = plt.subplots(nrows=2, ncols=3,
            subplot_kw={'adjustable': 'box-forced'})
    ax = axes.ravel()

    for blur_level in range(6):
        blurred = gaussian_filter(image, 6.0*blur_level, mode='nearest', cval=0)

        ax[blur_level].imshow(blurred, cmap='gray', interpolation='nearest')
        ax[blur_level].set_title(str(blur_level), fontsize=20)
    plt.show()

if __name__ == '__main__':
    draw_blur_levels()

