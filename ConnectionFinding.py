


from numpy.core.numeric import count_nonzero
import math
import time
import sys
import cv2
import os
from collections import Counter
import numpy as np
from PcbFunctions import *


font = cv2.FONT_HERSHEY_SIMPLEX

ImageName = "Board8.png"
Debugging_enable = False
Write_Enable = False
ICS_Introduced = True

img = cv2.imread(
    'assets/Example_images/Board_images/{}'.format(ImageName), cv2.IMREAD_COLOR)
if img is None:
    sys.exit("Could not read the image.")


if Debugging_enable:
    DEBUGGING_FILE1 = open("Debugging/DEBUGGING_FILE1.txt", "w")
    DEBUGGING_FILE1.write("~~~---START---~~~\n")
    


def PutOnTopBigBlack(image):
    
    s_img = image
    l_img = cv2.imread('black.png')
    
    x_offset = 300
    y_offset = 200
    l_img[y_offset:y_offset+s_img.shape[0],
          x_offset:x_offset+s_img.shape[1]] = s_img

    cv2.imshow("test", l_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return l_img

def DetectICsSilk(img):
    '''
    A function that detects an ICs silk traces (where an ICs should be placed) and hide it
    input: An image that the function should find the ics silk traces inside
    output: An image with those Silk traces removed, an array contining an x,y,w,h (rectangle) of where the traces are. the array has this format:
    [[x1,y1,w1,h1],
    [x2,y2,w2,h2]]
    this array should be exported down the line to a maybe a schamtic of somesort and an text finding algorthim which will read what the chip is
    and extract his schmatic data.'''

    IcsDetected = img.copy()
    copy = img.copy()
    
    
    BoardColor = GetDominotColor(img)
    
    
    
    
    
    lower_val = np.array([200, 200, 200])
    upper_val = np.array([255, 255, 255])
    
    mask = cv2.inRange(img, lower_val, upper_val)
    

    
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)

    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        print("looping over contours")
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        print("Approx: ", len(approx))
        area = cv2.contourArea(c)
        print("Area: ", area)
        
        if len(approx) == 4 and area > 100 and area < 70000:
            print("found ic")
            (x, y, w, h) = cv2.boundingRect(approx)
            
            
            
            cv2.rectangle(IcsDetected, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            cv2.rectangle(img, (x-2, y-2), (x+w, y+h),
                          (int(BoardColor[0]), int(BoardColor[1]), int(BoardColor[2])), -1)
            
            

    
    return img



try:
    EntireBoardPointsFile = open(
        "output/Files/PointsFileFor_{}.txt".format(ImageName), "r")
except OSError as e:
    sys.exit("Could not open the EBP points file. error: ", e)
EBP_String, EntireBoardPoints = GetPointsFromFile(EntireBoardPointsFile)


cv2.imshow('Original Image', img)
cv2.waitKey(0)

if ICS_Introduced:
    
    img = DetectICsSilk(img)
    cv2.imshow('Ics removed', img)
    cv2.waitKey(0)

###

cnts = GetContours(img)

CroppingMaskV2 = np.zeros_like(img)



dp = 1
min_dist = 0.1  
param_1 = 200
param_2 = 15
min_Radius = 0
max_Radius = 15


contour_counter = 0





pts = np.array([[1, 2], [3, 4]])        
pts = np.delete(pts, [0, 1], axis=0)    


ContourBoxPoints = np.array([[1, 2], [3, 4]])        

ContourBoxPoints = np.delete(ContourBoxPoints, [0, 1], axis=0)


starting_contour_number = 0



epsilon_value = 0.0009


Counter2 = 1


for c in cnts:
    
    Counter2 = 1
    
    
    
    approx = cv2.approxPolyDP(c, epsilon_value * cv2.arcLength(c, True), True)
    
    n = approx.ravel()
    i = 0
    for j in n:
        if(i % 2 == 0):
            x1 = n[i]
            y1 = n[i + 1]
            
            
            
            if(contour_counter > starting_contour_number):
                
                pts = np.append(pts, [[x1, y1]], axis=0)
        i = i + 1

    
    
    
    if(contour_counter > starting_contour_number):  
        
        rect = cv2.boundingRect(pts)
        
        x2, y2, w, h = rect
        print("x2 :", x2, " y2: ", y2)
        
        croped = img[y2:y2+h, x2:x2+w].copy()
        
        pts = pts - pts.min(axis=0)

        print(croped.shape[:2])
        
        CroppingMask = np.zeros(croped.shape[:2], np.uint8)
        #CroppingMask = np.full_like(croped, (255, 0, 0))

        
        
        cv2.drawContours(CroppingMask, [pts], -1,
                         (255, 255, 255), -1, cv2.LINE_AA)

        AntiCroppingMask = cv2.bitwise_not(CroppingMask)
        cv2.imshow("CroppingMask", CroppingMask)
        cv2.imshow("AntiCroppingMask", AntiCroppingMask)
        cv2.waitKey(0)


        dst = cv2.bitwise_and(croped, croped, mask=CroppingMask)
        cv2.imshow("dst1", dst)
        cv2.waitKey(0)

        
        
        dst_hsv = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)
        
        black_lo = np.array([0, 0, 0])
        black_hi = np.array([0, 0, 4])

        
        black_mask = cv2.inRange(dst_hsv, black_lo, black_hi)

        
        dst[black_mask > 0] = (29, 53, 5)

        cv2.imshow("dst2", dst)
        cv2.waitKey(0)

        #dst = cv2.bitwise_and(test, test, mask=AntiCroppingMask)
        #cv2.imshow("dst", dst)
        

        ContourBox = dst.copy()


        print("############\nCONTOUR NUMBER: {}\n############".format(contour_counter))

        
        ContourBoxPoints, ContourBox = DetectPointsV2(ContourBox)
        
        for Point in ContourBoxPoints:
            Point[0] = Point[0] + (x2)
            Point[1] = Point[1] + (y2)
        print("ContourBoxPoints:\n", ContourBoxPoints)
        print("EntireBoardPoints:\n", EntireBoardPoints)
        
        
        for Point1 in ContourBoxPoints:
            
            #Counter2 = 1
            for Point2 in EntireBoardPoints:
                
                print("COMPARING: ContourBoxPoints Point1: [{},{}]  EntireBoardPoints Point2: [{},{}]".format(
                    Point1[0], Point1[1], Point2[0], Point2[1]))
                if(math.isclose(Point1[0], Point2[0], rel_tol=0.02, abs_tol=0.0)):
                    print("x is close enough")
                    
                    if(math.isclose(Point1[1], Point2[1], rel_tol=0.02, abs_tol=0.0)):
                        print("y is close enough - a match")
                        print("Counter2 before: ", Counter2)
                        
                        INDEX1 = EBP_String.find(
                            "[{},{}]".format(Point2[0], Point2[1]))
                        print("index: ", INDEX1)
                        
                        INDEX1 = (EBP_String.find("]", INDEX1)) + 1
                        print("index: ", INDEX1)
                        
                        
                        try:
                            EBP_String = EBP_String[:INDEX1] + \
                                " connected to: ({},{})".format(
                                    ContourBoxPoints[Counter2][0], ContourBoxPoints[Counter2][1]) + EBP_String[INDEX1:]
                                                        
                            if Counter2 == 0:
                                Counter2 = 1
                            elif (Counter2 == 1):
                                Counter2 = 0

                            print(EBP_String)
                            break
                        except IndexError:
                            print("Error code 15: less than two points in contour.")

                print("No match.")

        
        ###

        ###

        
        bg = np.ones_like(croped, np.uint8)*255
        cv2.bitwise_not(bg, bg, mask=CroppingMask)
        dst2 = bg + dst

        
        pts = np.delete(pts, np.s_[:], axis=0)
        ContourBoxPoints = np.delete(ContourBoxPoints, np.s_[:], axis=0)

        
        '''
        cv2.imwrite('output\croped.png', croped)
        cv2.imwrite('output\mask.png', CroppingMask)
        cv2.imwrite('output\dst.png', dst)
        cv2.imwrite('output\dst2.png', dst2)

        '''


        cv2.imwrite('output/Images/{}croped.png'.format(
            contour_counter), croped)
        cv2.imwrite('output/Images/{}mask.png'.format(
            contour_counter), CroppingMask)
        cv2.imwrite('output/Images/{}dst.png'.format(
            contour_counter), dst)
        
        
        cv2.imwrite('output/Images/{}PointsInContourBox.png'.format(
            contour_counter), ContourBox)

    ####
    

    
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    
    first = cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
    

    
    cv2.drawContours(CroppingMaskV2, [c], -1, 255, -1)
    
    out = np.zeros_like(img)
    out[CroppingMaskV2 == 255] = img[CroppingMaskV2 == 255]

    #####

    contour_counter += 1
if(Write_Enable):
    EntireBoardPointsFileWithConnection = open(
        "Testing/PointsFileWithConnectionFor{}.txt".format(ImageName), "w")
    EntireBoardPointsFileWithConnection.write(EBP_String)
    EntireBoardPointsFileWithConnection.close()


print("NUM OF Contours: ", contour_counter)

'''

(y, x) = np.where(mask == 255)
(topy, topx) = (np.min(y), np.min(x))
(bottomy, bottomx) = (np.max(y), np.max(x))
out = out[topy:bottomy+1, topx:bottomx+1]
    '''

cv2.imshow('mask', out)
cv2.waitKey(0)

cv2.imshow('Objects Detected', img)
cv2.waitKey(0)

