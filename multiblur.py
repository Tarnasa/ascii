import numpy as np

from scipy.ndimage.filters import gaussian_filter
from collections import namedtuple

BlurScene = namedtuple('BlurScene', ('features', 'norm'))


def calculate_normalizer(scores):
    return np.sum(scores) / sum(scores.shape)

def score_rect(image, x, y, w, h, xs, ys, levels):
    features = np.zeros((levels, (h/ys+1) * (w/xs+1)))
    blur_factor = min(xs, ys)
    for blur_level in range(levels):
        blurred = gaussian_filter(image, blur_factor*blur_level, mode='nearest', cval=0)
        features[blur_level] = blurred[y:y+h:ys, x:x+w:xs].ravel()
    return BlurScene(features, calculate_normalizer(features))

def compare_scores(scene1, scene2):
    return np.sum(np.linalg.norm(scene1.features - scene2.features, ord=1, axis=1)) / (scene1.norm + scene2.norm)

def draw_blur_levels():
    import matplotlib.pyplot as plt
    from skimage import io

    image = io.imread('out/125.png')  # 36 for $, 79 for O

    fig, axes = plt.subplots(nrows=2, ncols=3,
            subplot_kw={'adjustable': 'box-forced'})
    ax = axes.ravel()

    for blur_level in range(6):
        blurred = gaussian_filter(image, 2.0*blur_level, mode='nearest', cval=0)

        ax[blur_level].imshow(blurred, cmap='gray', interpolation='nearest')
        ax[blur_level].set_title(str(blur_level), fontsize=20)
    plt.show()

if __name__ == '__main__':
    draw_blur_levels()

