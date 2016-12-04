"""
Generate an array of line segments representing the edges in the figure
"""

# From: http://stackoverflow.com/questions/38898554/is-that-possible-to-clean-a-contour-plot-in-skimage

import numpy as np
import matplotlib.pyplot as plt

from skimage import io, measure, morphology

from scipy import ndimage
from scipy.spatial.distance import pdist
from scipy.spatial import KDTree


def skel_to_graph(skel):
    """
    Transform skeleton into its branches and nodes, by counting the number
    of neighbors of each pixel in the skeleton
    """
    convolve_skel = 3**skel.ndim * ndimage.uniform_filter(skel.astype(np.float)) # 3x3 square mean
    neighbors = np.copy(skel.astype(np.uint8))
    skel = skel.astype(np.bool)
    neighbors[skel] = convolve_skel[skel] - 1
    edges = morphology.label(np.logical_or(neighbors == 2, neighbors ==1),
                            background=0)
    nodes = morphology.label(np.logical_and(np.not_equal(neighbors, 2),
                                            neighbors > 0), background=0)
    length_edges = np.bincount(edges.ravel())
    return nodes, edges, length_edges


def count_neighbors(binary):
    convolve = 3**binary.ndim * ndimage.uniform_filter(binary.astype(np.float)) # 3x3 sum
    neighbors = np.copy(binary.astype(np.uint8))
    binary = binary.astype(np.bool)
    neighbors[binary] = convolve[binary] - 1
    return neighbors


_neighbor_offsets = [
    (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1)
    ]
def skel_to_vectors(skel):
    """
    Transform the skeleton into pairs of adjacent points
    It is recommended to run approximate_polygon after this vectorization

    Method:
        Find its nearest neighbor out of all pixels composing the edges, make that an edge
        Continue with that neighbor until all pixels in that edge are part of the path
        Find a the node closest to the last edge pixel, add that to the path (It may be the same as the beginning node
        Do this again, but ignore the pixels of the edge already pathed
    """
    neighbors = count_neighbors(skel)
    # Pad with zeros so that neighbor check never goes outside matrix
    neighbors = np.pad(neighbors, ((1, 1), (1, 1)), 'constant', constant_values=[0, 0])
    edges = morphology.label(np.logical_or(neighbors == 2, neighbors ==1),
                             background=0)
    nodes = morphology.label(np.logical_and(np.not_equal(neighbors, 2),
                                            neighbors > 0), background=0)
    xn, yn = node_indices = np.nonzero(nodes)
    used_edges = set([0])
    paths = {}  # edge_id -> path
    for n in range(len(xn)):
        while True:
            x, y = xn[n], yn[n]
            # Find a neighboring edge_id that we haven't seen yet
            edge_id = None
            nx, ny = None, None
            for xo, yo in _neighbor_offsets:
                nx, ny = x+xo, y+yo
                edge_id = edges[nx, ny]
                if edge_id not in used_edges:
                    used_edges.add(edge_id)
                    break
            else:
                # We have exhausted this node's neighbors
                break
            # Follow the path of this edge to another node
            path = [[x-1, y-1], [nx-1, ny-1]]
            px, py = x, y
            x, y = nx, ny
            # Remove the edge label so that it isn't chosen as the next pixel's neighbor
            edges[px, py] = 0
            while True:
                nx, ny = None, None
                for xo, yo in _neighbor_offsets:
                    nx, ny = x+xo, y+yo
                    if edges[nx, ny] == edge_id:
                        break
                else:
                    # We've reached the end of this path
                    # If the endpoint is 3-or-more way node, then add that to the path too
                    for xo, yo in _neighbor_offsets:
                        nx, ny = x+xo, y+yo
                        if nodes[nx, ny]:
                            path.append([nx-1, ny-1])
                            break  # Technically not needed
                    break
                path.append([nx-1, ny-1])
                px, py = x, y
                x, y = nx, ny
                # Remove the edge label so that it isn't chosen as the next pixel's neighbor
                edges[px, py] = 0
            paths[edge_id] = np.array(path)
    return paths.values()


# Open image and threshold
image = io.imread('out/125.png')  # 36 for $, 79 for O
im = np.min(image, axis=2) < 20

# Skeletonize to get 1-pixel-thick lines
skel = morphology.skeletonize(im)
nodes, edges, length_edges = skel_to_graph(skel)
paths = skel_to_vectors(skel)
#print(repr(paths))

# blurred = 3**skel.ndim * ndimage.uniform_filter(skel.astype(np.float))
# contours = measure.find_contours(blurred, 0.5)
# #print(repr(contours))
# simpler = measure.approximate_polygon(contours[0], 4)
# print(repr(simpler))

# Visualize results
#plt.figure(figsize=(10, 10))
fig, ax = plt.subplots()
ax.imshow(im, cmap='gray', interpolation='nearest')
#paths[1] = measure.approximate_polygon(paths[1], 2)
for path in paths:
    path = measure.approximate_polygon(path, 1)
    ax.plot(path[:, 1], path[:, 0], linewidth=2)

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
#plt.contour(edges > 0, [0.5], colors='red', linewidths=[2])
#ax.axis('off')
#ax.tight_layout()
plt.show()



"""
# Construct some test data
x, y = np.ogrid[-np.pi:np.pi:100j, -np.pi:np.pi:100j]
r = np.sin(np.exp((np.sin(x)**3 + np.cos(y)**2)))

# Find contours at a constant value of 0.8
contours = measure.find_contours(r, 0.99)

# Display the image and plot all contours found
fig, ax = plt.subplots()
ax.imshow(r, interpolation='nearest', cmap=plt.cm.gray)

for n, contour in enumerate(contours):
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()
"""

# TODO: Graph lengths of segments in image, compare to lengths of segments in characters, warn if large discrepency

