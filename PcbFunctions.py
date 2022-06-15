
import math
import numpy as np
import cv2
import sys
import io
from skidl import search,show



# images of points, used for image matching
RectPointRight_img = cv2.imread(
    'assets/Example_images/Board_points/rectPoint.png', cv2.IMREAD_COLOR)
CircPoint_img = cv2.imread(
    'assets/Example_images/Board_points/CircPoint.png', cv2.IMREAD_COLOR)

pointImages = [RectPointRight_img, CircPoint_img]

def get_ordered_list(points, x, y):
   points.sort(key = lambda p: (p.x - x)**2 + (p.y - y)**2)
   return points

def GetPointsFromFile(File):
    '''
    a function thats returning the file as a string and numpy array containing all the points
    expects this format:
    Point: [x1,y1]
    Point: [x2,y2]
    Point: [x3,y3]
    returns that as a string and a numpy array in this format:
    [[x1,y1],
    [x2,y2],
    [x3,y3]]
    '''
    # moving to a string so i could manipulatie it
    EBP_String = File.read()
    # variable that stores the entire board x,y coordinates
    # dummy array initialisation
    EntireBoardPoints = np.array([[1, 2], [3, 4]])
    # clearing the array for inputs
    EntireBoardPoints = np.delete(EntireBoardPoints, [0, 1], axis=0)

    # getting the data from the string into numpy array
    end_bracket_index = 0
    while(end_bracket_index != len(EBP_String)):  # not really necessary
        start_bracket_index = EBP_String.find("[", end_bracket_index)
        middle_index = EBP_String.find(",", end_bracket_index)
        # if not the +1 it would give the same position
        end_bracket_index = EBP_String.find("]", end_bracket_index+1)
        # print("start_bracket_index: ", start_bracket_index, " middle_index: ",
        #      middle_index, " end_bracket_index: ", end_bracket_index)
        if (start_bracket_index == -1):  # if not found
            break
        else:
            EntireBoardPoints = np.append(
                EntireBoardPoints, [[int(EBP_String[(start_bracket_index+1): (middle_index)]),
                                     int(EBP_String[(middle_index+1): (end_bracket_index)])]], axis=0)  # not sure why but this form works

    return EBP_String, EntireBoardPoints


def GetContours(img):
    '''
    One of the core functions of this whole algorithm.
    returns Contours.
    '''
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #cv2.imshow('HSV Image', hsv)
    # cv2.waitKey(0)
    hue, saturation, value = cv2.split(hsv)
    #cv2.imshow('Saturation Image', saturation)
    # cv2.waitKey(0)
    retval, thresholded = cv2.threshold(
        saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #cv2.imshow('Thresholded Image', thresholded)
    # cv2.waitKey(0)
    medianFiltered = cv2.medianBlur(thresholded, 5)
    #cv2.imshow('Median Filtered Image', medianFiltered)
    # cv2.waitKey(0)
    cnts, hierarchy = cv2.findContours(
        medianFiltered, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return cnts


def GetDominotColor(img):
    '''
    a function that returns the most dominot color in an image.\n
    returns a bgr format of the color: [B,G,R]
    '''
    colors, count = np.unique(
        img.reshape(-1, img.shape[-1]), axis=0, return_counts=True)
    return colors[count.argmax()]



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

def DetectPointsV2(image, Debugging_Enabled = True):
    '''
    ## version 2 of DetectingCircles.\n
    This function takes an image and returns a numpy 3-diminisonal array contining all points.
    the array would look like this:\n
    [[x1,y1],\n
    [x2,y2],\n
    [x3.y3]]\n
    Got from => https://stackoverflow.com/questions/60637120/detect-circles-in-opencv second answer.
    @image an image that the algorithm should find the points in

    '''
    copy = image.copy()
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 11)
    # a varibale that display's how much points was found, if 1 then that means i didn't get the other one so resort to diffrent methods
    Num_Points_Found = 0
    # it also captures the rectangles points
    thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    if Debugging_Enabled:
        cv2.imshow('blur', blur)
        cv2.imshow('thresh', thresh)

    # dummy array intsilation
    BoardPointsArray = np.array([[1, 2], [3, 4]])
    # clearing the array for inputs
    BoardPointsArray = np.delete(BoardPointsArray, [0, 1], axis=0)

    # finding rectangles
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        print("Approx: ", len(approx))
        area = cv2.contourArea(c)
        print(area)
        if len(approx) == 4 and area > 50 and area < 200:
            (x, y, w, h) = cv2.boundingRect(approx)
            #ar = w / float(h)
            #cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
            cv2.rectangle(copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
            BoardPointsArray = np.append(
                BoardPointsArray, [[int(((w)/2) + x), int(((h)/2) + y)]], axis=0)
            Num_Points_Found += 1

    #cv2.imshow('RectanglesDetectedByV2', copy)

    # finding circles

    # Morph open
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)
    #cv2.imshow('opening', opening)

    # Find contours and filter using contour area and aspect ratio
    cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)

    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        area = cv2.contourArea(c)
        # print(area)
        if len(approx) > 5 and area > 100 and area < 500000:
            ((x, y), r) = cv2.minEnclosingCircle(c)
            cv2.circle(copy, (int(x), int(y)), int(r), (36, 255, 12), 2)
            BoardPointsArray = np.append(
                BoardPointsArray, [[int(x), int(y)]], axis=0)
            Num_Points_Found += 1
    if Debugging_Enabled:
        cv2.imshow('Both_Rec&Circs_DetectedByV2', copy)
        print("Num_Points_Found before image matching: ", Num_Points_Found)
        print(BoardPointsArray)

    # if only found one point or less, try to find others using image matching
    if Num_Points_Found < 2:
        print("NOTE: on this run, I found less than two points -> trying to find others using image matching!")
        
        # Because were using image matching, we need to try each of our images of how the pcb points looks like
        # return x,y,w,h of the image of the point inside the bigger image
        for pointImage in pointImages:
            FoundPoint = Template_matching(image, pointImage, 0.81, Debugging_Enabled, BoardPointsArray)
            if FoundPoint:

                cv2.rectangle(copy, (FoundPoint[0], FoundPoint[1]), (FoundPoint[0] +
                                                        FoundPoint[2], FoundPoint[1]+FoundPoint[3]), (0, 0, 255), 2)
                
                if Debugging_Enabled: print("found a FoundPoint point at: x1,y1: {},{}; x2,y2: {},{}".format(
                    FoundPoint[0], FoundPoint[1], FoundPoint[0] + FoundPoint[2],  FoundPoint[1]+FoundPoint[3]))
                
                # entering the middle point of the FoundPoint - for best accuarcy
                # [ [ w / 2 + x, h / 2 + y ] ]
                BoardPointsArray = np.append(
                    BoardPointsArray, [[int((FoundPoint[2]/2)+FoundPoint[0]), int((FoundPoint[3]/2)+FoundPoint[1])]], axis=0)

                Num_Points_Found += 1
            elif Debugging_Enabled: print("NOTE: Image matching returned None.")
        

        if Num_Points_Found < 2:
            print("Error - Image matching Cannot find all points :(")

        if Num_Points_Found > 2:
            print("Error: 3 or more points found.")
            print(f"BoardPointsArray: {BoardPointsArray} ")
        
    else:
        print("2 points found!")

    print("Num_Points_Found after image matching: ", Num_Points_Found)
    #cv2.imshow('thresh', thresh)
    #cv2.imshow('opening', opening)
    #cv2.imshow('CirclesDetectedByV2', image)
    return BoardPointsArray, copy


def Template_matching(img, Img_Point, DesValue = 0.81, Debug_Enable = False, AlreadyFoundPoints = None):
    '''
    A function that looks for an image inside of another img.\n
    If the threshold is less than DesValue, the function would return None

    Input: the image that The point should be inside it, The image of how the point should look like,
    A value used to filter false positives.
    Output: The x,y,w,h of where the Img_Point is inside img OR a None if threshold not meeted. The function always returns something.
    @img the image that the algorthim should find Img_Point inside
    @Img_Point the image that should be found inside img
    @DesValue a value used to filter false positives
    @Debug_Enable show verbose printing and image processing
    @AlreadyFoundPoints optinal parameter to avoid getting points that are already detected
    '''
    

    '''
    # a variable to store the matched cordienates instances of the rect of where
    # the Matched image was found. [x,y,w,h]
    #MatchingRectCordArray = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    # clearing the array for inputs
    #MatchingRectCordArray = np.delete(MatchingRectCordArray, np.s_[:], axis=0)

    #img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Img_Point_gray = cv2.cvtColor(Img_Point, cv2.COLOR_BGR2GRAY)

    w, h = Img_Point.shape[:2]

    result = cv2.matchTemplate(img_gray, Img_Point_gray,
                               cv2.TM_CCOEFF_NORMED)

    (yCoords, xCoords) = np.where(result >= 0.8)

    for (x, y) in zip(xCoords, yCoords):
        # draw the bounding box on the image
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
        MatchingRectCordArray = np.append(
            MatchingRectCordArray, [[x, y, x + w, y + h]], axis=0)

    #w, h, _ = Img_Point.shape[::-1]
    res = cv2.matchTemplate(img_gray, Img_Point_gray, cv2.TM_SQDIFF_NORMED)
    threshold = 0.9
    loc = np.where(res >= threshold)
    i = 0
    for pt in zip(*loc[::-1]):
        if(i > 3):
            break
        print("!!!")
        cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
        MatchingRectCordArray = np.append(
            MatchingRectCordArray, [[pt[0], pt[1], pt[0] + w,  pt[1] + h]], axis=0)
        i += 1
    
    '''

    if(Debug_Enable):
        cv2.imshow("TMStartImg", img)
        #cv2.imshow("imgpoint", Img_Point)

    #result = cv2.matchTemplate(Img_Point, img, cv2.TM_SQDIFF_NORMED)
    result = cv2.matchTemplate(Img_Point, img, cv2.TM_CCOEFF_NORMED)
    if(Debug_Enable):
        cv2.imshow('O_TM_Template', result)

    # We want the minimum squared difference
    (mn, maxVal, mnLoc, maxLoc) = cv2.minMaxLoc(result)

    conf = result.max()

    if(Debug_Enable):
        print("maxVal: ", maxVal)
        print("conf: ", conf)

    # senstivity check
    if(maxVal > DesValue):
        # Draw the rectangle:
        # Extract the coordinates of our best match
        x, y = maxLoc

        # Step 2: Get the size of the template. This is the same size as the match.
        h, w = Img_Point.shape[:2]

        # If we get something in alreadyFoundPoints, Lets throw those points in a check
        # as we got an area, just take the middle point and compare that.
        if AlreadyFoundPoints is not None:
            for Point in AlreadyFoundPoints:
                if isThosePointsTheSame( (w / 2 + x), (h / 2 + y), Point[0], Point[1] ):
                    return None

        if Debug_Enable:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            print(f"Threshold meeted: boxInTemplateMatching: x1,y1: {x},{y}; x2,y2: {x+w},{y+h}")
            # Display the original image with the rectangle around the match.
            cv2.imshow('O_TM', img)
            # print(MatchingRectCordArray)
            cv2.waitKey(0)

        return x, y, w, h
    else:
        if Debug_Enable:
            print("Threshold not meeted")
            # for threshold
            #cv2.waitKey(0)
        return None

# Math functions
def calculateDistance(x1,y1,x2,y2):
    """
    This function calculates the distance between two points.
    TODO: input validation;
    """
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

"""
def sortPointsByDistToRefPoint(refPoint, Points):
    
    This function sorts an array of points based on their distance to a reference point.
    
    return np.array(sorted(Points,key=lambda point:calculateDistance(refPoint[0],refPoint[1],*point)))
"""

def sortPointsByDistToRefPoint2(refPoint, Points):
    """
    This function sorts an array of points based on their distance to a reference point.
    """
    return sorted(Points,key=lambda point:calculateDistance(refPoint.x,refPoint.y,*[point.x,point.y]))


def isThosePointsTheSame(x1: int, y1: int, x2: int, y2: int, rel_tol: float = 0.15, abs_tol: float = 10.0) -> bool :
    """
    This function return True/False based on if two input points are close enough to be called
    The same.
    """
    #print(f"are those the same? x1,y1: {x1},{y1} x2,y2: {x2},{y2}")
    # checking if x is the same;  ~10 abs pixels margin
    if(math.isclose(x1, x2, rel_tol=rel_tol, abs_tol=abs_tol)):
        # x is close enough
        # check for y; ~10 pixels abs margin of error
        if(math.isclose(y1, y2, rel_tol=rel_tol, abs_tol=abs_tol)):
            return True

    return False


# IC info functions

def GetAmountOfPins(IcPinInfo):
    """
    This function takes skidl's format of ic information and returns the total
    pins of that ic. takes this format:
    ['ATtiny841-SS ():', 'Pin None/1/VCC/POWER-IN', 'Pin None/10/PA3/BIDIRECTIONAL', 'Pin None/11/PA2/BIDIRECTIONAL', 'Pin None/12/PA1/BIDIRECTIONAL', 'Pin None/13/AREF/PA0/BIDIRECTIONAL', 'Pin None/14/GND/POWER-IN', 'Pin None/2/XTAL1/PB0/BIDIRECTIONAL', 'Pin None/3/XTAL2/PB1/BIDIRECTIONAL', 'Pin None/4/~{RESET}/PB3/BIDIRECTIONAL', 'Pin None/5/PB2/BIDIRECTIONAL', 'Pin None/6/PA7/BIDIRECTIONAL', 'Pin None/7/PA6/BIDIRECTIONAL', 'Pin None/8/PA5/BIDIRECTIONAL', 'Pin None/9/PA4/BIDIRECTIONAL']
    """
    i = 0
    for pin in IcPinInfo:
        if "Pin" in pin:
            i = i + 1
    return i


def ICimageToSkidl(IC_image, reader, MinICchars: int = 4, Debugging_Enable = False):
    """
    This function takes an image of an IC and anaylse it to extract the name and pinout from skidl search algorithm.
    @IC_image image of an IC
    @reader EasyOCR reader object
    """

    TextResults = reader.readtext(IC_image)
    candidates = FilterResults(TextResults)
    chipAngle = 0
    
    if Debugging_Enable:
        cv2.imshow("IC_image",IC_image)
        print(f"TextResults: {TextResults}")
        print(f"candidates: {candidates}")
        cv2.waitKey(0)
    
    # for faster testing with Board11.png
    #candidates = [([[22, 12], [120, 12], [120, 38], [22, 38]], '74HC595', 0.8238449835122339)]
    
    # if 0, then rotate 90 ccw
    if len(candidates) == 0:
        IC_image_ROT90ccw = cv2.rotate(IC_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        chipAngle = 90

        TextResults = reader.readtext(IC_image_ROT90ccw)
        candidates = FilterResults(TextResults)

        if Debugging_Enable:
            cv2.imshow("IC_image",IC_image_ROT90ccw)
            print(f"TextResults: {TextResults}")
            print(f"candidates: {candidates}")
            cv2.waitKey(0)

        # if still 0, rotate 180
        if len(candidates) == 0:
            IC_image_ROT180 = cv2.rotate(IC_image, cv2.ROTATE_180)
            chipAngle = 180

            if Debugging_Enable:
                cv2.imshow("IC_image",IC_image_ROT180)
                TextResults = reader.readtext(IC_image_ROT180)
                candidates = FilterResults(TextResults)
                print(f"TextResults: {TextResults}")
                print(f"candidates: {candidates}")
        
            if len(candidates) == 0: return "Unknown IC name", "Unknown IC desc", None

    # loop over candidates, search every one with skidl search(); if there is a match just return that

    # because skidl search function actually outputs to stdout, i have to redirect it
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout
    output = []
    for candidate in candidates:
        SearchResult = search(str(candidate[1]))
        results = "".join(str(new_stdout.getvalue()).split()).split('...')

        # reversing list to detect junk output faster
        for result in reversed(results):
            if "Searching" in result: break
            else:
                # example: '74xx.kicad_sym:74HC595()' => [1] = 74xx.kicad_sym [2] => 74HC595
                output.append(result[:result.find(':')])
                output.append(result[(result.find(':') + 1):-2])
    
    # revert back
    sys.stdout = old_stdout

    if Debugging_Enable: print(output)
    
    # validating output
    i = 0
    while i < len(output):
        show_Message = show(output[i], output[i+1])
        if "ERROR" and "WARNING" not in show_Message:
            print(f"Found IC: ICname: {output[i+1]} ICdesc: {output[i]}")
            # return Chip name, Chip description, angle
            return output[i+1], output[i], chipAngle
        i += 2
    
    return "Unknown IC name", "Unknown IC desc", None


def FilterResults(Results, MinICchars = 4, minThreshold = 0.5):
    candidates = []
    for Result in Results:
        # First filter all with len < 4    => There is propaly no IC with that few chars
        # Second filter out too low threshold scores
        if len(Result[1]) > MinICchars and Result[2] > minThreshold:
            candidates.append(Result)
    return candidates

###############################################################################
# All of those methods uses SIFT OR SUFT and dont work as of 13/12 in opencv-contrib 4.3.x
###############################################################################

def Feature_matching2(img, Img_Point):
    '''
    A function that looks for an image inside of another img.\n
    Note: Super dumb algorithm that just sweeps through the image trying to find the perfect match.
    Should not be used for anything sensitive.
    Input: the image that The point should be inside it, The image of how the point should look like,
    And a value used to filter false positives.
    Output: The x,y,w,h of where the Img_Point is inside img. Or a -1 if threshold not meeted. The function always returns something.
    @img the image that the algorthim should find Img_Point inside
    @Img_Point the image that should be found inside img
    @DesValue a value used to filter false positives - a threshold
    '''
    cv2.imshow("test", img)
    cv2.imshow("imgpoint", Img_Point)

    # a variable to store the matched cordienates instances of the rect of where
    # the Matched image was found. [x,y,w,h]
    MatchingRectCordArray = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    # clearing the array for inputs
    MatchingRectCordArray = np.delete(MatchingRectCordArray, np.s_[:], axis=0)

    #img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Img_Point_gray = cv2.cvtColor(Img_Point, cv2.COLOR_BGR2GRAY)

    # -- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
    minHessian = 400
    detector = cv2.xfeatures2d_SURF.create(hessianThreshold=minHessian)
    keypoints1, descriptors1 = detector.detectAndCompute(img, None)
    keypoints2, descriptors2 = detector.detectAndCompute(Img_Point, None)

    # -- Step 2: Matching descriptor vectors with a FLANN based matcher
    # Since SURF is a floating-point descriptor NORM_L2 is used
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    knn_matches = matcher.knnMatch(descriptors1, descriptors2, 2)

    # -- Filter matches using the Lowe's ratio test
    ratio_thresh = 0.7
    good_matches = []
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
    # -- Draw matches
    img_matches = np.empty(
        (max(img.shape[0], Img_Point.shape[0]), img.shape[1]+Img_Point.shape[1], 3), dtype=np.uint8)
    cv2.drawMatches(img, keypoints1, Img_Point, keypoints2, good_matches,
                    img_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Display the original image with the rectangle around the match.
    cv2.imshow('outputImageMatching', img_matches)
    print(MatchingRectCordArray)
    cv2.waitKey(0)


def match_images(img1, img2, img1_features=None, img2_features=None):
    """Given two images, returns the matches"""
    detector = cv2.SURF(3200)
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    if img1_features is None:
        kp1, desc1 = detector.detectAndCompute(img1, None)
    else:
        kp1, desc1 = img1_features

    if img2_features is None:
        kp2, desc2 = detector.detectAndCompute(img2, None)
    else:
        kp2, desc2 = img2_features

    # print 'img1 - %d features, img2 - %d features' % (len(kp1), len(kp2))

    raw_matches = matcher.knnMatch(desc1, trainDescriptors=desc2, k=2)
    kp_pairs = filter_matches(kp1, kp2, raw_matches)
    return kp_pairs


def filter_matches(kp1, kp2, matches, ratio=0.75):
    """Filters features that are common to both images"""
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append(kp1[m.queryIdx])
            mkp2.append(kp2[m.trainIdx])
    kp_pairs = zip(mkp1, mkp2)
    return kp_pairs


# Match Diplaying

def draw_matches(window_name, kp_pairs, img1, img2):
    """Draws the matches"""
    mkp1, mkp2 = zip(*kp_pairs)

    H = None
    status = None

    if len(kp_pairs) >= 4:
        p1 = np.float32([kp.pt for kp in mkp1])
        p2 = np.float32([kp.pt for kp in mkp2])
        H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)

    if len(kp_pairs):
        explore_match(window_name, img1, img2, kp_pairs, status, H)


def explore_match(win, img1, img2, kp_pairs, status=None, H=None):
    """Draws lines between the matched features"""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1 + w2), np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1 + w2] = img2
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    if H is not None:
        corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        reshaped = cv2.perspectiveTransform(corners.reshape(1, -1, 2), H)
        reshaped = reshaped.reshape(-1, 2)
        corners = np.int32(reshaped + (w1, 0))
        cv2.polylines(vis, [corners], True, (255, 255, 255))

    if status is None:
        status = np.ones(len(kp_pairs), np.bool_)
    p1 = np.int32([kpp[0].pt for kpp in kp_pairs])
    p2 = np.int32([kpp[1].pt for kpp in kp_pairs]) + (w1, 0)

    green = (0, 255, 0)
    red = (0, 0, 255)
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            col = green
            cv2.circle(vis, (x1, y1), 2, col, -1)
            cv2.circle(vis, (x2, y2), 2, col, -1)
        else:
            col = red
            r = 2
            thickness = 3
            cv2.line(vis, (x1 - r, y1 - r), (x1 + r, y1 + r), col, thickness)
            cv2.line(vis, (x1 - r, y1 + r), (x1 + r, y1 - r), col, thickness)
            cv2.line(vis, (x2 - r, y2 - r), (x2 + r, y2 + r), col, thickness)
            cv2.line(vis, (x2 - r, y2 + r), (x2 + r, y2 - r), col, thickness)
    vis0 = vis.copy()
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            cv2.line(vis, (x1, y1), (x2, y2), green)

    cv2.imshow(win, vis)


def Feature_matching3(img, Img_Point):
    '''
    A function that looks for an image inside of another img.\n
    Note: Super dumb algorithm that just sweeps through the image trying to find the perfect match.\n
    Should not be used for anything sensitive.
    Input: the image that The point should be inside it, The image of how the point should look like,\n
    A value used to filter false positives.
    Output: The x,y,w,h of where the Img_Point is insdie img. The function always returns something.
    @img the image that the algorthim should find Img_Point inside
    @Img_Point the image that should be found inside img
    @DesValue a value used to filter false positives
    '''
    cv2.imshow("test", img)
    cv2.imshow("imgpoint", Img_Point)

    # a variable to store the matched cordienates instances of the rect of where
    # the Matched image was found. [x,y,w,h]
    MatchingRectCordArray = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    # clearing the array for inputs
    MatchingRectCordArray = np.delete(MatchingRectCordArray, np.s_[:], axis=0)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Img_Point_gray = cv2.cvtColor(Img_Point, cv2.COLOR_BGR2GRAY)

    # Initiate ORB detector
    orb = cv2.ORB_create()
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img_gray, None)
    kp2, des2 = orb.detectAndCompute(Img_Point_gray, None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1, des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)
    # Draw first 10 matches.
    img3 = cv2.drawMatches(img_gray, kp1, Img_Point_gray, kp2,
                           matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Display the original image with the rectangle around the match.
    cv2.imshow('outputImageMatching', img3)
    print(MatchingRectCordArray)
    cv2.waitKey(0)


def create_blank(width, height, bgr_color=(0, 0, 0)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((height, width, 3), np.uint8)
    # Since OpenCV uses BGR, convert the color first
    #color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = bgr_color

    return image
