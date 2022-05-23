"""
Please run THIS program first!

This program is responsible for detecting EBP (entire board points) and writing that data into a file
as well as showing an x and y coordinates for a clicked pixel.
"""

import numpy as np
import cv2
import sys
from collections import Counter
import os
from numpy.lib.function_base import copy
from PcbFunctions import *

print("~~~---START - Detecting Points---~~~")


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
        cv2.putText(
            img,
            (f'{str(x)},' + str(y)),
            (x, y),
            font,
            0.7,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


        cv2.circle(img, (x, y), 2, (0, 0, 0), -1)

        cv2.imshow('original image', img)

###########
# Golbal Variables
###########

# variable that stores the entire board x,y coordinates
EntireBoardPoints = np.array([[1, 2], [3, 4]])
# clearing the array for inputs
EntireBoardPoints = np.delete(EntireBoardPoints, [0, 1], axis=0)

# SET BEFORE USE!
ImageName = "Board9.png"
write_to_file = True

# Write all the points to a file for further analysis by ConnectionFinding.py
if write_to_file:
    try:
        POINTS_FILE = open(f"output/Files/PointsFileFor_{ImageName}.txt", "w")
        POINTS_FILE.write("EntireBoardPoints program:\n")
    except OSError as e:
        sys.exit("Could not open directory to output the Points file to. error: ", e)




# cv2.namedWindow("output", cv2.WINDOW_FREERATIO)

img = cv2.imread(
    f'assets/Example_images/Board_images/{ImageName}', cv2.IMREAD_COLOR
)

if img is None:
    sys.exit("Could not read the image.")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rows = gray.shape[0]  # 93.625
original_img = img.copy()

# detecting circles V2
EntireBoardPoints, img = DetectPointsV2(img)

print(EntireBoardPoints)
cv2.imshow('original image', img)

# write to a file all the points x,y
if(write_to_file):
    for PoInt in EntireBoardPoints:
        POINTS_FILE.write("\nPoint: [{},{}]".format(PoInt[0], PoInt[1]))

    POINTS_FILE.close()

# handling clicked on a pixel in the image to show the x and y cords
cv2.setMouseCallback('original image', click_event)

# input handler
input = cv2.waitKey(0)

while input not in [27, -1]:  # input != esc
    print(input)

    if(input == 114):  # input == R; ==> Reset Pic
        img = original_img.copy()
        cv2.imshow("board image", img)

    input = cv2.waitKey(0)  # waits for any input


cv2.destroyAllWindows()

print("~~~---FINISH---~~~")