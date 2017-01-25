#!/usr/bin/env python
import cv2
import sys
import linedetect
import time
import convert
import ocr_svm
import numpy as np
import json
from utility import generate_alphabet
import resource
from math import ceil
from multiprocessing import Pool, JoinableQueue, Process
from asciiplayer import play

def ascii_frame(image, alphabet, svm, resolution):
    """
    Applies preprocessing and converts a single image frame to ascii art
    """
    lines = linedetect.lines(image, resolution)
    image = lines.astype(np.float32)
    m = np.max(image)
    image /= m
    if len(image.shape) == 3:
        image = 1 - np.min(image, axis=2)
    else:
        image = 1 - image

    art = convert.convert(image, alphabet, svm, 1)
    return art

def multiprocessing_ascii_frame(data):
    """
    Used by multiprocessing.Pool which only allows for a single argument
    """
    return ascii_frame(*data)

def convert_movie(movie_path, resolution=(3200, 1800), max_ram=6000000, realtime=False):
    """
    Converts a movie to ascii art
    """
    resolution = (1600, 900)

    alphabet = generate_alphabet('out/*.png', 1)
    svm = ocr_svm.svm_single(1)
    vidcap = cv2.VideoCapture(movie_path)
    success = True
    i = 0
    chunk = 0


    total_frames = int(vidcap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    last_played = None

    fps = int(vidcap.get(cv2.cv.CV_CAP_PROP_FPS))
    width = int(vidcap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))

    print("File: {}  Resolution: {}x{}  Framerate: {}fps".format(movie_path, width, height, fps))
    print("Target resolution: {}x{}".format(resolution[0], resolution[1]))
    
    frames = []
    seek = []
    frames_per_chunk = total_frames
    total_chunks = 1

    player, frame_queue = None, None

    if realtime:
        frames_per_chunk=400
        total_chunks = int(ceil(total_frames/float(frames_per_chunk)))
        frame_queue = JoinableQueue()
        player = Process(target=play, args=(frame_queue,))
        player.start()


    while success:
        if not realtime:
            sys.stdout.write("\r" + " " * 100)
            sys.stdout.flush()

        pool = Pool(6)
        jobs = []
        images = []
        chunk += 1

        while success:

            success,image = vidcap.read()
            if not success:
                break

            images.append((image, alphabet, svm, resolution))
            seek.append(str(vidcap.get(cv2.cv.CV_CAP_PROP_POS_MSEC)))

            if not realtime:
                sys.stdout.write("\rReading... {}/{} ({}%)".format(i + 1, total_frames, int((i + 1)/float(total_frames) * 100)))
                sys.stdout.flush()
            i += 1

            if chunk == 1 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > max_ram/2.0:
                if frames_per_chunk == total_frames:
                    frames_per_chunk = i
                    total_chunks = int(ceil(total_frames/float(frames_per_chunk)))
                break

            if i > chunk * frames_per_chunk:
                break

        if not success:
            vidcap.release()

        results = pool.map_async(multiprocessing_ascii_frame, images)

        pool.close()

        total = results._number_left

        while True:
            remaining = results._number_left
            if not realtime:
                sys.stdout.write("\rConverting... {}/{} ({}%)".format(total - remaining, total, int((total - remaining)/float(total) * 100)))
                sys.stdout.write(" in chunk {}/{} (Overall: {}%)".format(chunk, total_chunks, int((chunk - 1 + (total - remaining)/float(total))/ (total_chunks) * 100)))
                sys.stdout.flush()
            if results.ready(): 
                break

            time.sleep(0.2)

        pool.join()

        fr = results.get()
        frames.extend(fr)

        if realtime:
            for i in range(len(fr)):
                frame_queue.put((float(seek[-len(fr) + i]), fr[i]))
            
            

        del pool, results, jobs, images


    print(" Done!")

    if realtime:
        player.join()
    else:
        asciimovie = dict(zip(seek, frames))
        with open(movie_path + ".ascii", "w") as file:
            json.dump({'frames':asciimovie, 'fps':fps}, file)
            print("Saved as {}.ascii".format(movie_path))
    return asciimovie

if __name__ == "__main__":
    asciimovie = convert_movie(sys.argv[1], realtime=False)
    #for seektime in sorted(asciimovie.keys(), key=float):
    #    print(asciimovie[seektime])
    #    time.sleep(1/float(fps))