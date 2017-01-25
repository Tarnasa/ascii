#!/usr/bin/env python
import time
import os

from multiprocessing import JoinableQueue, Process
from threading import Thread
import subprocess
import sys
import json

def play(frames):
    """
    Plays an ascii art movie
    Arguments:
        frames: A Queue contaning the frames to be played
    """
    dropped = 0
    start = None
    playing = True
    buffer_time = 0

    while True:
        try:
            msg = frames.get()
            if msg == "END":
                break
            if not playing:
                continue
            seek, frame = msg
            if not start:
                start = time.time()
            wait_time = seek/1000.0 - (time.time() - start)

            if wait_time >= 0: 
                print(frame)
                time.sleep(wait_time)
            else:
                dropped += 1
                start = time.time() - (seek / 1000.0)
        except KeyboardInterrupt, Exception:
            playing = False

    print("Dropped frames: {}".format(dropped))

if __name__ == "__main__":
    
    frame_queue = JoinableQueue()
    player = Process(target=play, args=(frame_queue,))
    player.start()

    with open(sys.argv[1]) as file:
        shortname = "".join(sys.argv[1].split(".")[0])
        movie = json.load(file)
        
    subprocess.Popen(["ffplay", "-nodisp", "-autoexit", "{}.mp4".format("".join(sys.argv[1].split(".")[0]))], stdout=subprocess.PIPE, stderr = subprocess.STDOUT)

    for seek in sorted(movie["frames"], key=float)[10:]:#, None, float(movie["fps"]))
        frame_queue.put((float(seek), str(movie["frames"][str(seek)])))

    frame_queue.put("END")
    
    try:
        player.join()
    except KeyboardInterrupt, Exception:
        pass

