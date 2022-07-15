
"""
This program can be used as a BOM (“Bill of Materials”) extractor for a given pcb image with some tuning required.
The output is a csv file containing every component x,y,x+w,y+h coordinates which can be fed to predict.py in PCB-CD to get components label.
The background of the pcb image should be cropped first, the values below are for the RPI3B_Bottom example. 

"""

# Resources:
# https://stackoverflow.com/questions/63138270/opencv-change-to-nearest-color-in-image
# https://stackoverflow.com/questions/63015856/python-opencv-color-shape-based-detection
# https://stackoverflow.com/questions/62891917/how-to-change-the-colour-of-an-image-using-a-mask
# https://stackoverflow.com/questions/58051025/how-to-remove-color-from-image

import sys
import cv2
import numpy as np

outputEnable = False
img = cv2.imread('PCB-Component-Detection/pcb_wacv_2019/RPI3B_Bottom/RPI3B_Bottom.jpg', cv2.IMREAD_COLOR)
if img is None:
    sys.exit("[EE] Couldn't read image. Please check file integrity/path.")

# Cropping background
        #   x, y      x    y
region = [[245,140], [1405,880]]
img = img[region[0][1] : region[1][1] , region[0][0] : region[1][0]]

def ResizeWithAspectRatio(image, width=None, height=800, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

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


cv2.imshow("img", img)
cv2.setMouseCallback('img', click_event)

ORIGINAL_IMG = img.copy()

cv2.waitKey(0)


# removing Lables like R50 C3 IC12;

# BGR values of white lables
lower_val = np.array([170, 170, 140])
upper_val = np.array([220, 240, 220])

boardColor = np.array([56, 115, 10])

# for future: HSV
#lower_val = np.array([0,0,59])
#upper_val = np.array([0,0,100])
#imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# creating the Mask - selecting white lables
mask = cv2.inRange(img, lower_val, upper_val)
cv2.imshow("mask", ResizeWithAspectRatio(mask))

_, mask = cv2.threshold(mask, thresh=100, maxval=255, type=cv2.THRESH_BINARY)
#mask = cv2.GaussianBlur(mask,(1,1),0)


#kernel = np.ones((2,1),np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=1)
#erosion = cv2.erode(mask,kernel,iterations = 1)
dilation = cv2.dilate(opening,kernel,iterations = 2)

cv2.imshow("dilation", ResizeWithAspectRatio(dilation))

# Could be useful
#img = cv2.inpaint(img, dilation, 2, cv2.INPAINT_NS)

# filling each label with boardColor, effectively disguising it.
img[(dilation>60)] = boardColor
#img = cv2.bitwise_and(img, img, mask=mask)

cv2.imshow("img",ResizeWithAspectRatio(img))


cv2.waitKey(0)


# creating offset for selecting boardcolor
lower_val = boardColor - 35
lower_val[lower_val < 0] = 0

upper_val = boardColor + 30

# create the Mask - selecting boardcolor
mask = cv2.inRange(img, lower_val, upper_val)
# inverse mask
mask = 255-mask

_, mask = cv2.threshold(mask, thresh=170, maxval=255, type=cv2.THRESH_BINARY)
cv2.imshow("mask", ResizeWithAspectRatio(mask))


kernel = np.ones((1,1),np.uint8)
erosion = cv2.erode(mask,kernel,iterations = 3)
dilation = cv2.dilate(erosion,kernel,iterations = 1)
#dilation = 255 - dilation
cv2.imshow("dilation", ResizeWithAspectRatio(dilation))

# Creating white image to store output to
result = np.zeros_like(img)
result = 255 - result

#result[(dilation==255)] = img[dilation]
result = cv2.bitwise_and(img,img, mask=dilation)


print(f"img shape: {img.shape} result shape: {result.shape}")

cv2.imshow("result", result)
cv2.waitKey(0)


failedCheck = 0
rects = []

cnts = cv2.findContours((cv2.threshold(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY), 127, 255, 0))[1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    #print("looping over chips")
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    #print("Approx: ", len(approx))
    area = cv2.contourArea(c)
    #print("Area: ", area)
    
    if area > 2 and area < 50000:
        
        (x, y, w, h) = cv2.boundingRect(approx)
        
        #print("contour at: {},{} to: {},{}".format(x,y,(x+w),(y+h)))
        
        rects.append([x,y,w,h])
        # show what ics got detected
        #cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 1)
    else:
        failedCheck += 1

print(f"Failed check: {failedCheck}")

def groupRect(rectarray, rectthreshold=1, eps=0.1):
    """
    Groups similar rectangles together in rectangle array \n
    Input : Rectangle Array \n
    Output : Rectangle Array without split parts
    """
    results = cv2.groupRectangles(np.concatenate((rectarray, rectarray)), rectthreshold,eps=eps)[0]
    results = [[group] for group in results]
    return np.array(results)

rects = groupRect(rects)
print(f"Number of components found: {len(rects)}")

if outputEnable:
    csvFile = open("Image2schematic/output/Files/BOM.csv", 'a')

for rect in rects:
    rect = rect[0]
    cv2.rectangle(result, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (0,0,255), 1)

    # Saving output for further anaylsis by PCB-CD

    if outputEnable:
        csvFile.write(f"{rect[0]},{rect[1]},{rect[0]+rect[2]},{rect[1]+rect[3]}\n")

if outputEnable: csvFile.close()

#cv2.imshow("result", result)


# restore the orginal color for better visul look
temp_mask = cv2.inRange(result,np.array([0,0,0]), np.array([1,1,1]))
result[temp_mask != 0] = ORIGINAL_IMG[temp_mask != 0]

cv2.imshow("result", result)

cv2.waitKey(0)


