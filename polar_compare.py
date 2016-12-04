"""
Compare two graymaps by using the multiple log-polar method described in:
https://pdfs.semanticscholar.org/d42c/069253c40a4795b51afffe53d02ff7f844cd.pdf
"""

from scipy.ndimage.filters import gaussian_filter
from scipy.misc import imresize

import numpy as np
import math

from collections import namedtuple, defaultdict

Scene = namedtuple('Scene', ('image', 'scores', 'offsets', 'blurred', 'padding'))



def fit(w, h, subw, subh):
    """
    Calculates the best width and height to scale the original rectangle
    to which minimizes change in width + height.  I think.
    """
    new_w = subw * int(w / float(subw) + 0.5)
    new_h = subh * int(h / float(subh) + 0.5)
    return new_w, new_h


def score_all(image, cell_width, cell_height, r_sections=5, theta_sections=12):
    """ image is should be a grayscale image with a black background """
    h, w = image.shape
    # Scale image to perfectly fit cell size
    w, h = fit(w, h, cell_width, cell_height)
    image = imresize(image, (h, w), interp='bilinear')
    # Pad image so that we never walk off the array
    r = int(min(cell_width, cell_height) / 2)
    padding = r + 1
    # Pad with black (lack of ink) because image will be presented to user that way
    # TODO: Validate that this is actually the best choice
    image = np.pad(image, ((padding, padding), (padding, padding)),
            'constant', constant_values=0)
    # Calculate offsets needed by score_cell
    offsets = generate_offsets_per_bin(r, r_sections, theta_sections)
    x_offsets, y_offsets = offsets
    total_bins = r_sections * theta_sections
    # Blur image so that log-polar bins can be sampled by looking at one point
    blurred = gaussian_filter(image, 3.0, mode='nearest', cval=0)
    # Calculate scores of all subsections
    x_cells, y_cells = w / cell_width, h / cell_height
    scores = np.zeros((y_cells, x_cells, cell_height*cell_width*total_bins/4))
    for yi, y in enumerate(range(padding, h+padding, cell_height)):
        for xi, x in enumerate(range(padding, w+padding, cell_width)):
            scores[yi, xi] = score_cell(blurred, cell_width, cell_height,
                    x, y, x_offsets, y_offsets).ravel()
    return Scene(image, scores, offsets, blurred, padding)


def score_cell_old(im, w, h, xoff, yoff, offsets):
    # TODO use numpy arrays
    bins = []
    for y in range(1, h, 2):
        xbins = []
        for x in range(1, w, 2):
            xbins.append(im[offsets[1] + y+yoff, offsets[0] + x+xoff])
        bins.append(xbins)
    return np.sum(np.array(bins))


def score_cell(im, w, h, xoff, yoff, x_offsets, y_offsets):
    points = []
    for y in range(1, h, 2):
        xpoints = []
        for x in range(1, w, 2):
            xpoints.append(score_point(im, x+xoff, y+yoff, x_offsets, y_offsets))
            """
            bins = []
            for xoffs, yoffs in zip(x_offsets, y_offsets):
                bins.append(np.sum(im[yoffs+y+yoff, xoffs+x+xoff]))
            # TODO: Apply +xoff once to entire x_offsets
            xpoints.append(bins)
            """
        points.append(xpoints)
    return np.array(points)


def score_point(im, x, y, x_offsets, y_offsets):
    bins = []
    for xoffs, yoffs in zip(x_offsets, y_offsets):
        bins.append(np.average(im[yoffs+y, xoffs+x]))
    return bins



def generate_offsets_old(r, r_sections=5, theta_sections=12):
    """
    Generates a numpy array like:
    [[-1, 0, 1, 0], [0, 1, 0, -1]]
    representing the points [[-1, 0], [0, 1], [1, 0], [0, -1]]
    For the given log-polar paramters
    """
    offsets = np.zeros((r_sections * theta_sections, 2), dtype=np.float)
    i = 0
    for ri in range(1, r_sections+1):
        for ti in range(0, theta_sections):
            radius = ri * (r / r_sections)
            # TODO: Vectorize sin and cos operations
            theta = (math.pi * ti * 2) / theta_sections
            offsets[i][0] = radius * math.cos(theta)
            offsets[i][1] = radius * math.sin(theta)
            i += 1
    offsets = np.asarray(np.around(offsets), dtype=int)
    return offsets.transpose()


def generate_offsets_per_bin(r, r_sections=5, theta_sections=12):
    # This funcion is not meant to be called multiple times
    per_theta = [0 for _ in range(theta_sections)] # TODO Remove
    per_radii = [0 for _ in range(r_sections)]
    total_bins = r_sections * theta_sections
    x_offsets = [list() for _ in range(total_bins)]
    y_offsets = [list() for _ in range(total_bins)]
    for y in range(-r, r+1):
        for x in range(-r, r+1):
            if x or y:
                # log a / log b = log_b(a)
                radius_index = int(math.log((x*x + y*y)**0.5, r) * r_sections)
                theta = math.atan2(y, x) + math.pi  # atan2 goes from -pi to pi
                theta = (theta + (2*math.pi/theta_sections)/2) % (2*math.pi)
                theta_index = int((theta / (2*math.pi)) * theta_sections)
                if theta_index == theta_sections:  # Turn closed to open interval
                    theta_index -= 1
                #print(x, y, radius_index, theta_index) # TODO Remove
                if radius_index < r_sections:
                    bin_index = radius_index*theta_sections + theta_index
                    x_offsets[bin_index].append(x)
                    y_offsets[bin_index].append(y)
                    per_theta[theta_index] += 1
                    per_radii[radius_index] += 1
            else:
                # Add center point to some bin
                x_offsets[0].append(x)
                y_offsets[0].append(y)
    for bin_index in range(total_bins):
        x_offsets[bin_index] = np.array(x_offsets[bin_index], dtype=np.int)
        y_offsets[bin_index] = np.array(y_offsets[bin_index], dtype=np.int)
    print('theta sections', per_theta)
    print('radius sections', per_radii)
    return x_offsets, y_offsets


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from skimage import io

    og = io.imread('out/125.png')  # 36 for $, 79 for O
    #image = gaussian_filter(image, 3.0, mode='nearest')
    image = 255 - np.min(og, axis=2)
    w, h = 16, 22
    scene = score_all(image, w, h)

    v, u = scene.image.shape
    colored = np.zeros((v, u, 4), dtype=np.uint8)
    padding = scene.padding
    for y in range(v):
        for x in range(u):
            colored[y, x] = [scene.image[y, x], scene.image[y, x], scene.image[y, x], 255]

    # Draw bins
    xspb, yspb = generate_offsets_per_bin(33)
    xoff, yoff = 23, 40
    print('shape', scene.blurred.shape)
    scores = score_point(scene.image, xoff+padding, yoff+padding, xspb, yspb)
    for bin, (xs, ys) in enumerate(zip(xspb, yspb)):
        for x, y in zip(xs, ys):
            #colored[y+yoff+padding, x+xoff+padding, 0:3] = (bin * 255) / 60
            colored[y+yoff+padding, x+xoff+padding, 0:2] = scores[bin]


    # Tint each cell by it's score
    """
    yscores, xscores = scene.scores.shape
    for yi in range(yscores):
        for xi in range(xscores):
            for yo in range(h):
                for xo in range(w):
                    y = yi*h+yo+padding
                    x = xi*w+xo+padding
                    colored[y, x, 0] = min(255, colored[y, x, 0] + scene.scores[yi, xi])
    """

    """
    w, h, _ = image.shape
    offsets = generate_offsets(min(w, h) / 2)
    x, y = 25, 25
    image[offsets[0] + x, offsets[1] + y] = [255, 0, 0, 255]
    """

    fig, ax = plt.subplots()
    ax.imshow(colored, cmap='gray', interpolation='nearest')
    plt.show()

