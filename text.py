#!/usr/bin/env python2
"""
For rendering text as bitmaps
"""

"""
import logging

import pygame
import pygame.freetype

pygame.init()

font = pygame.freetype.Font(file='/usr/share/fonts/TTF/DejaVuSansMono.ttf')

def render_text_as_surface(text, size=16):
    return font.render(text, fgcolor=(255,255,255,255), bgcolor=(0,0,0,255), size=size)[0]

def render_text_as_features(text, size=16):
    surface = render_text_as_surface(text, size)

s = render_text_as_surface('Pancake')
pygame.image.save(s, 'test.png')
"""


from subprocess import Popen

s_command = """
    convert \
        -background white \
        -fill black \
        -font {FONT} \
        -stroke black \
        -pointsize {POINTSIZE} \
        label:{STR} "{OUT}/{c}.png"
"""


def render_text_to_file(text, filename, font='DejaVu-Sans-Mono', size=12):
    if type(size) == int:
        size = str(size)
    if text == '\\':
        text = '\\\\'
    command = [
        'convert',
        '-background', 'white',
        '-fill', 'black',
        '-font', font,
        '-stroke', 'black',
        '-pointsize', size,
        "label:{}".format(text), filename
    ]
    print(' '.join(command))
    Popen(command)

if __name__ == '__main__':
    #import argparse
    #parser = argparse.ArgumentParser()
    for code in range(33, 127):
        render_text_to_file(chr(code), 'out/' + str(code) + '.png', size=90)

