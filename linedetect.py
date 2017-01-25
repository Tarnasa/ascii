#!/usr/bin/env python
import numpy as np
import glob
import cv2
import sys

def lines(image, resolution=(1600,900)):
    """
    Detect edges in an image and expand them by blurring 
    so they can be easier to recognize by the converter
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    wide = cv2.Canny(blurred, 15, 50)

    lines = wide
    cv2.bitwise_not(lines, lines)
    lines = rescale(lines, resolution[0], resolution[1])

    for i in range(8):
        lines = cv2.GaussianBlur(lines, (5,5), 0)
    ret, lines = cv2.threshold(lines, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return lines

def blank(image):
    """
    Returns a black image of the same size as the input image
    """
    height, width = image.shape
    output = np.zeros((height, width, 3), np.uint8)
    output[:] = (255, 255, 255)
    return output

def rescale(image, x, y):
    """
    Scales an image to the given x, y
    """
    height, width = image.shape
    fx = x/float(width)
    fy = y/float(height)
    return cv2.resize(image, (0, 0), fx=fx, fy=fy)