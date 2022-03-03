

import numpy as np
import cv2
import sys
from collections import Counter
import os
# TODO: switch to using arg parse
# import argparse
from numpy.lib.function_base import copy
from PcbFunctions import *

print("~~~---START---~~~")

# args handler
# parser = argparse.ArgumentParser(description='This script take an image and outputs a text file containing all the x,y cords of the on board points.')

# Mouse click handler TODO: move to pcb or OpenCV functions
def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:

        # displaying the coordinates
        # on the Shell

        colorsB = img[y, x, 0]
        colorsG = img[y, x, 1]
        colorsR = img[y, x, 2]
        colors = img[y, x]

        print("Coordinates: X: ", x, "Y: ", y, "[ B: ",
              colorsB, "G: ", colorsG, "R: ", colorsR, " ]")
        # print("BRG Format: ", colors)
        # print("B: ", colorsB, "G: ", colorsG, "R: ", colorsR)

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + ',' +
                    str(y), (x, y), font,
                    0.7, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.circle(img, (x, y), 2, (0, 0, 0), -1)
        if(len(points) < 2):
            points.append((x, y))
        else:
            points.clear()
            points.append((x, y))

        global pts
        pts = np.array(points, np.int32)
        # cv2.polylines(img, [pts], False, (255, 0, 0))

        cv2.imshow('orginial image', img)

###########
# Golbal Variables
###########

# variable that stores 2 current clicked position for line drawing..
points = [] # TODO: check for usefulenss

# variable that stores the entire board x,y cordinates
EntireBoardPoints = np.array([[1, 2], [3, 4]])      # dummy array intsilation TODO: make an actual init
# clearing the array for inputs
EntireBoardPoints = np.delete(EntireBoardPoints, [0, 1], axis=0)

# SET BEFORE USE!
ImageName = "Board8.png"
write_to_file = True

# Write all the points to a file for further analysis by T_ConnectionFinding.py
if(write_to_file):
    try:
        POINTS_FILE = open(
            "output/Files/PointsFileFor_{}.txt".format(ImageName), "w")
        POINTS_FILE.write("EntireBoardPoints program:\n")
    except OSError as e:
        sys.exit("Could not open directory to output the Points file to. error: ", e)




# cv2.namedWindow("output", cv2.WINDOW_FREERATIO)

img = cv2.imread('assets/Example_images/Board_images/{}'.format(ImageName), cv2.IMREAD_COLOR)
if img is None:
    sys.exit("Could not read the image.")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rows = gray.shape[0]  # 93.625
original_img = img.copy()

# detecting circles V2
EntireBoardPoints, img = DetectPointsV2(img)

print(EntireBoardPoints)
cv2.imshow('orginial image', img)

# write to a file all the points x,y
if(write_to_file):
    for PoInt in EntireBoardPoints:
        POINTS_FILE.write("\nPoint: [{},{}]".format(PoInt[0], PoInt[1]))

    POINTS_FILE.close()

# handling clicked on a pixel in the image to show the x and y cords
cv2.setMouseCallback('orginial image', click_event)

# input handler
input = cv2.waitKey(0)

while(input != 27 and input != -1):  # input != esc
    print(input)

    if(input == 114):  # input == R; ==> Reset Pic
        img = original_img.copy()
        cv2.imshow("board image", img)

    input = cv2.waitKey(0)  # waits for any input


cv2.destroyAllWindows()

print("~~~---FINISH---~~~")