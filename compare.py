from polar import score_all
from multiblur import score_rect, compare_scores
import numpy as np

def compare(scene1, scene2):
    diff = scene1.scores[0, 0] - scene2.scores[0, 0]
    diff = np.sum(np.linalg.norm(diff, ord=1, axis=2))
    diff /= scene1.grayness + scene2.grayness
    return diff
    

if __name__ == '__main__':
    from skimage import io
    import glob

    w, h = 55, 106
    scenes = dict()
    filenames = glob.glob('out/*.png')
    filenames = [f for f in filenames if '4' in f and '1' not in f]
    filenames.append('test.png')
    print(filenames)
    #filenames = ['out/49.png', 'out/47.png']
    for filename in filenames:
        print('reading', filename)
        image = io.imread(filename)
        if len(image.shape) == 3:
            image = 255 - np.min(image, axis=2)
        else:
            image = 255 - image
        #scenes[filename] = score_all(image, w, h)
        scenes[filename] = score_rect(image, 0, 0, w, h, w/8, h/8, 5)

    target = scenes['test.png']
    comparisons = []
    for filename, scene in scenes.items():
        #diff = scene.scores[0, 0] - target.scores[0, 0]
        #diff = np.sum(np.linalg.norm(diff, ord=1, axis=2))
        #diff /= scene.grayness + target.grayness
        #compairsons.append((diff, filename))
        diff = compare_scores(scene, target)
        comparisons.append((diff, filename))
    for score, filename in sorted(comparisons, key=lambda p: p[0]):
        print(score, filename)

