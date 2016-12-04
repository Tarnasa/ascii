from polar import score_all
import numpy as np

def compare(scene1, scene2)
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
        scenes[filename] = score_all(image, w, h)

    target = scenes['test.png']
    compairsons = []
    for filename, scene in scenes.items():
        diff = scene.scores[0, 0] - target.scores[0, 0]
        diff = np.sum(np.linalg.norm(diff, ord=1, axis=2))
        diff /= scene.grayness + target.grayness
        compairsons.append((diff, filename))
    for score, filename in sorted(compairsons):
        print(score, filename)

