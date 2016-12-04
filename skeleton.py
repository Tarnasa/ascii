#!/usr/bin/env python2
"""
Test out scikit-image's skeletonization
"""

import sys

from skimage.morphology import skeletonize
from skimage import data

import matplotlib.pyplot as plt

import numpy as np

horse = data.horse()
image = 255 - horse
image = np.max(image, axis=2) / 255. + 0.5
image = image.astype(int)
print(repr(image))
#sys.exit(0)

skeleton = skeletonize(image)
print(repr(skeleton))

# display results
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4),
                         sharex=True, sharey=True,
                         subplot_kw={'adjustable': 'box-forced'})

ax = axes.ravel()

ax[0].imshow(horse, cmap=plt.cm.gray)
ax[0].axis('off')
ax[0].set_title('original', fontsize=20)

ax[1].imshow(skeleton, cmap=plt.cm.gray)
ax[1].axis('off')
ax[1].set_title('skeleton', fontsize=20)

fig.tight_layout()
plt.show()