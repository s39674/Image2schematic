
"""
Please run this program SECOND!

this program finds connections between points.
This program also associate every point with an IC that's connected to it.
All the data then gets writen to a new points file.
"""

import math
import sys
import cv2
import os
import numpy as np
from PcbFunctions import *
from skidl import search,show
from point import *
from chip import *
from PrintedCircutBoard import *

MyPCB = PrintedCircutBoard()


# Please change this value according to your image name
ImageName = "Board8.png"
# Debug mode allows you to see the image processing more clearly
Debugging_Enable = False
# Choose to write the Connections to the PointsFile or not. 
Write_Enable = True
# If there are no chips/integrated circuits, the process could be a lot faster.
ICS_Introduced = True
IC_detectTest = True
# If above is true, please provide the name of the IC that works with skidl search function,
# IC identification is a future feature,
# TODO: add t
IcName = ['MCU_Microchip_ATtiny','ATtiny841-SS']

# stores chips corrds for later use- associting each pin with its corrsponding x,y cords
IcCords = np.array([[1, 2, 3, 4]])        # dummy array intsilation
IcCords = np.delete(IcCords, 0, axis=0)    # clearing the array for inputs
print(IcCords)

img = cv2.imread(
    'assets/Example_images/Board_images/{}'.format(ImageName), cv2.IMREAD_COLOR)
if img is None:
    sys.exit("Could not read the image.")

"""
# for future use
if Debugging_Enable:
    DEBUGGING_FILE1 = open("Debugging/DEBUGGING_FILE1.txt", "w")
    DEBUGGING_FILE1.write("~~~---START---~~~\n")
"""


def DetectICsSilk(img, Threshold_AreaMin = 80, Threshold_AreaMax = 70000):
    '''
    This function detects an ICs silk traces (where an ICs should be placed) and hides it, that
    needed for a better trace finding. This function also populates IcCords for future analysis.
    input: An image that the function should find the ics silk traces inside
    output: An image with those Silk traces removed
    '''
    # making the IcCords a global so i modify it and the modifcation will stay after the function
    # TODO: make the function return this
    global IcCords
    IcsDetected = img.copy()
    
    # a variable that stores the bgr color value of the board. Its used for hiding the silk trace with a rectangle whose color is this variable
    # this assusmes the most used color is the color of the board!
    BoardColor = GetDominotColor(img)
    
    
    
    
    # setting lower and upper limit of the color, should be white for silk traces
    lower_val = np.array([170, 170, 170])
    upper_val = np.array([255, 255, 255])
    # Threshold the bgr image to get only that range of colors
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
        
        if len(approx) == 4 and area > Threshold_AreaMin and area < Threshold_AreaMax:
            
            (x, y, w, h) = cv2.boundingRect(approx)
            
            print("found ic at: {},{} to: {},{}".format(x,y,(x+w),(y+h)))
            IcCords = np.append(IcCords, [[int(x), int(y), int(x+w),int(y+h)]], axis=0)

            MyPCB.addChip( chip( point(int(x), int(y)), point(int(x+w),int(y+h)), IcName[0], IcName[1] , ConnectedToPCB=MyPCB))
            
            # show what ics got detected
            cv2.rectangle(IcsDetected, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # hiding that silk trace with a rectangle whose color is the same as the entire board
            cv2.rectangle(img, (x-2, y-2), (x+w, y+h),
                          (int(BoardColor[0]), int(BoardColor[1]), int(BoardColor[2])), -1)
            
    return img


# get the EntireBoardPoints from the file
try:
    EntireBoardPointsFile = open(
        "output/Files/PointsFileFor_{}.txt".format(ImageName), "r")
except OSError as e:
    sys.exit("Could not open the EBP points file. error: ", e)

EBP_String, EntireBoardPoints = GetPointsFromFile(EntireBoardPointsFile)

for EBP in EntireBoardPoints:
    MyPCB.addPoint(point(EBP[0],EBP[1]))

cv2.imshow('Original Image', img)
cv2.waitKey(0)

if ICS_Introduced:
    # removing Ics silk traces so it will not corrupt the output
    img = DetectICsSilk(img)
    cv2.imshow('Ics removed', img)
    cv2.waitKey(0)
    
    print("Ics at:", IcCords)

    """
    ClosePinPoints = np.array([[1, 2], [3, 4]])        # dummy array intsilation
    ClosePinPoints = np.delete(ClosePinPoints, [0, 1], axis=0)    # clearing the array for inputs
    TempClosePinPoints = np.array([[1, 2], [3, 4]])
    TempClosePinPoints = np.delete(TempClosePinPoints, [0, 1], axis=0)
    """

    ClosePinPoints = []
    TempClosePinPoints = []

    ### BEFORE i write the connections (because i want to rewrite the file), i want to loop over EBP
    ### and check (by seeing if the x or y (should be determined by the oriantion of the chip)
    ### if it's close enough to the outer right or left line of the chip) to see if a prticular pin is an IC pin,
    ### if it is i want to replace the in in the points file with something like this: lm336 Pin[1] Vcc [x,y]
    ### and after the connection is found it should say this: 
    ### lm336 Pin[1] Vcc [x,y] connected to: hc07 Pin[4] A0 [x,y] 
    ### or if the other pin is a regular point: lm336 Pin[1] Vcc [x,y] connected to: [x,y] 

    ### pseudo code:
    ### loop EBP, check if the x is close to the x of ic
    ### if it is, check the y: check if: Y of upper line - 20 < Y of point < Y of lower line + 20
    ### if it is, put it in an array
    ### replace every element of that array with the distance of that point to the right most point of the ic
    ### sort the array to get the nearst point first => that becomes pin 1
    ### Repeat for the left side of the array. For a four sided IC, the procces should be the same just with inverted y and x procedures.
    #### For future: i don't even need to calculate the distance as it already sort them i just need to reverse it.

    for Chip in MyPCB.chips:
        Chip.printInfo()
        print(f"right line x of Ic: {Chip.DownRightMostPoint.x}")

        #RightMostPointOfIc = [Chip[2],Chip[1]]
        #RightMostPointOfIc = [Chip.DownRightMostPoint.x, Chip.UpLeftMostPoint.y]
        RightMostPointOfIc = point(Chip.DownRightMostPoint.x, Chip.UpLeftMostPoint.y)

        #LeftMostPointOfIC = [Chip[0],Chip[1]]
        #LeftMostPointOfIC = [Chip.UpLeftMostPoint.x, Chip.UpLeftMostPoint.y]
        LeftMostPointOfIC = Chip.UpLeftMostPoint

        # Looping on NC points to save time; for the First IC there is no benefit
        # NOT writing Point.Connected to chip, i'll do it after the arragment
        for NCpoint in MyPCB.ReturnAllNCpoints():
            # check x of point and x of right line of ic
            if(math.isclose(NCpoint.x, Chip.DownRightMostPoint.x, rel_tol=0.2, abs_tol=10)):
                # TODO: generalize those values
                if( (Chip.UpLeftMostPoint.y - 20) < NCpoint.y and NCpoint.y < (Chip.DownRightMostPoint.y + 20) ):
                    # If the point is actually close to where i suspect an IC points will be:
                    #ClosePinPoints = np.append(ClosePinPoints, [[int(NCpoint.x), int(NCpoint.y)]], axis=0)
                    ClosePinPoints.append(NCpoint)
                elif Debugging_Enable: print("Failed y - right")
            elif Debugging_Enable: print("Failed x - right")

        # TODO: Finally fixed this algorithm to work with the classes, can now just set the IC pins at the end
        ClosePinPoints = sortPointsByDistToRefPoint2(RightMostPointOfIc, ClosePinPoints)
        
        # now the same process for the left side; Looping on NC points to save time
        for NCpoint in MyPCB.ReturnAllNCpoints():
            # check x of point vs x of left line of ic
            if(math.isclose(NCpoint.x, Chip.UpLeftMostPoint.x, rel_tol=0.2, abs_tol=10)):
                if( (Chip.UpLeftMostPoint.y - 20) < NCpoint.y and NCpoint.y < (Chip.DownRightMostPoint.y + 20) ):
                    #TempClosePinPoints = np.append(TempClosePinPoints, [[int(NCpoint.x), int(NCpoint.y)]], axis=0)
                    TempClosePinPoints.append(NCpoint)
                elif Debugging_Enable: print("Failed y - left")
            elif Debugging_Enable: print("Failed x - left")
        
        # Now i got the right side points in ClosePinPoints, and left side in TempClosePinPoints, because i want all
        # of the right points to appear first, i concatenate it
        # IF TempClosePinPoints has no points, it cannot concatenate it (idk why)
        if (len(TempClosePinPoints) > 0):
            TempClosePinPoints = sortPointsByDistToRefPoint2(LeftMostPointOfIC, TempClosePinPoints)
            # now that we got the right order of pins set, the left order is just appened at the end
            # That way i get the right order for the pinout 
            #ClosePinPoints = np.concatenate((ClosePinPoints, TempClosePinPoints))
            ClosePinPoints = ClosePinPoints + TempClosePinPoints # => [right1,right2,left1,left2]
            
        Chip.ConnectPINS(ClosePinPoints)

        # for a 4 sided IC, should be the same process just with the x and y inverted: x,y = y,x

        #for poi in ClosePinPoints:
        #    print(f"[{poi.x}, {poi.y}]")


        CurrentICqueryResult = list(filter(bool, [str.strip() for str in (str(show(Chip.IcName,Chip.IcDescription))).splitlines()]))
        #print(CurrentICqueryResult)
        if Write_Enable and IC_detectTest:
            with open("output//Files//PointsFileFor_{}.txt".format(ImageName), 'r') as file:
                filedata = file.read()

            i = 1
            for ClosePinPoint in ClosePinPoints:
                #print(ClosePinPoint)
                # the skidl ic query format is not perfect. it starts at pin 1 and goes to pin 10,11,12 and so on.
                # this takes care of it.
                while f"/{i}/" not in CurrentICqueryResult[i]:
                    CurrentICqueryResult.append(CurrentICqueryResult.pop(i))
                                                                             # "ATtiny841-SS ():" => "ATtiny841-SS"; "Pin None/1/VCC/POWER-IN" => "1/VCC/POWER-IN"
                filedata = filedata.replace(f'Point: [{ClosePinPoint.x},{ClosePinPoint.y}]', f'{CurrentICqueryResult[0][:-4]} | {CurrentICqueryResult[i][9:]} | [{ClosePinPoint.x},{ClosePinPoint.y}]')

                i = i + 1

            #print(filedata)
            with open(f"output//Files//PointsFileFor_{ImageName}.txt", 'w') as file:
                file.write(filedata)

cnts = GetContours(img)

CroppingMaskV2 = np.zeros_like(img)


contour_counter = 0

# an array to store the contour x,y cords
pts = np.array([[1, 2], [3, 4]])        
pts = np.delete(pts, [0, 1], axis=0)    


ContourBoxPoints = np.array([[1, 2], [3, 4]])        

ContourBoxPoints = np.delete(ContourBoxPoints, [0, 1], axis=0)


starting_contour_number = 0


# used to calculate the epsilon for approxPolyDP() => https://stackoverflow.com/questions/62274412/cv2-approxpolydp-cv2-arclength-how-these-works
epsilon_value = 0.0009


Counter2 = 1

# looping on each contour - which is actually the wire, and marking the two points inside it as "connected"
for c in cnts:
    
    Counter2 = 1
    
    
    
    approx = cv2.approxPolyDP(c, epsilon_value * cv2.arcLength(c, True), True)
    
    n = approx.ravel()
    i = 0
    for j in n:
        if(i % 2 == 0):
            x1 = n[i]
            y1 = n[i + 1]
            
            
            # the first contour is actually the background, so just ignore that.
            if(contour_counter > starting_contour_number):
                
                pts = np.append(pts, [[x1, y1]], axis=0)
        i = i + 1

    
    
    
    if(contour_counter > starting_contour_number):  
        
        rect = cv2.boundingRect(pts)
        
        x2, y2, w, h = rect
        print("x2 :", x2, " y2: ", y2)
        
        croped = img[y2:y2+h, x2:x2+w].copy()
        
        pts = pts - pts.min(axis=0)

        #print(croped.shape[:2])
        
        CroppingMask = np.zeros(croped.shape[:2], np.uint8)
        #CroppingMask = np.full_like(croped, (255, 0, 0))

        
        
        cv2.drawContours(CroppingMask, [pts], -1,
                         (255, 255, 255), -1, cv2.LINE_AA)

        AntiCroppingMask = cv2.bitwise_not(CroppingMask)

        if Debugging_Enable:
            cv2.imshow("CroppingMask", CroppingMask)
            cv2.imshow("AntiCroppingMask", AntiCroppingMask)
            cv2.waitKey(0)


        dst = cv2.bitwise_and(croped, croped, mask=CroppingMask)

        if Debugging_Enable:
            cv2.imshow("dst1", dst)
            cv2.waitKey(0)

        
        
        dst_hsv = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)
        
        black_lo = np.array([0, 0, 0])
        black_hi = np.array([0, 0, 4])

        
        black_mask = cv2.inRange(dst_hsv, black_lo, black_hi)

        
        dst[black_mask > 0] = (29, 53, 5)

        if Debugging_Enable:
            cv2.imshow("dst2", dst)
            cv2.waitKey(0)

        #dst = cv2.bitwise_and(test, test, mask=AntiCroppingMask)
        #cv2.imshow("dst", dst)
        

        ContourBox = dst.copy()


        if Debugging_Enable: print("############\nCONTOUR NUMBER: {}\n############".format(contour_counter))

        
        ContourBoxPoints, ContourBox = DetectPointsV2(ContourBox, Debugging_Enable)
        
        # After i got all the board points which are inside the Contour box, i need to pair those with EntireBoardPoint
        # According to this formula: X (EntireBoardPoint) = X (In ContourBox) + X (Where box starts), same with Y
        # X (where box starts) = x2
        # Y (where box starts) = y2

        for Point in ContourBoxPoints:
            Point[0] = Point[0] + (x2)
            Point[1] = Point[1] + (y2)
        
        if Debugging_Enable:
            print("ContourBoxPoints:\n", ContourBoxPoints)
            print("EntireBoardPoints:\n", EntireBoardPoints)
        
        
        for Point1 in ContourBoxPoints:
            
            #Counter2 = 1
            for Point2 in EntireBoardPoints:
                
                if Debugging_Enable:
                    print("COMPARING: ContourBoxPoints Point1: [{},{}]  EntireBoardPoints Point2: [{},{}]".format(
                        Point1[0], Point1[1], Point2[0], Point2[1]))
                
                if isThosePointsTheSame(Point1[0], Point1[1], Point2[0],Point2[1], rel_tol=0.02, abs_tol=0.0):
                        
                    INDEX1 = EBP_String.find(
                        "[{},{}]".format(Point2[0], Point2[1]))
                    
                    INDEX1 = (EBP_String.find("]", INDEX1)) + 1
                    
                    # https://stackoverflow.com/questions/5254445/how-to-add-a-string-in-a-certain-position
                    try:
                        EBP_String = EBP_String[:INDEX1] + \
                            " connected to: ({},{})".format(
                                ContourBoxPoints[Counter2][0], ContourBoxPoints[Counter2][1]) + EBP_String[INDEX1:]

                        Counter2 = int(not Counter2)
                        #print(EBP_String)
                        break
                    except IndexError:
                        print("Error code 15: less than two points in contour.")

                #print("No match.")

        
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

        # exporting all.
        if Debugging_Enable:
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
        "output//Files//PointsFileWithConnectionFor{}.txt".format(ImageName), "w")
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

cv2.imshow('Objects Detected', img)
cv2.waitKey(0)

