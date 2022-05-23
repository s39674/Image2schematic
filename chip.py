
from point import *

class chip:
    
    IcName = None
    # an array to store all pointers to the pcb points that are connected to this ic
    pins = []
    # up left most cornor
    UpLeftMostPoint = point()
    # down right most cornor
    DownRightMostPoint = point()
    
    def __init__(self, UpLeftMostPoint = None, DownRightMostPoint = None, IcName = None, pins: list[point] = None):
        if UpLeftMostPoint: self.UpLeftMostPoint = UpLeftMostPoint
        else: self.UpLeftMostPoint = point()

        if DownRightMostPoint: self.DownRightMostPoint = DownRightMostPoint
        else: self.DownRightMostPoint = point()

        if pins: 
            self.pins = pins
            # looping over recived pins to set the ConnectedToIC variable
            for point in pins:
                point.ConnectedToChip = self
        if IcName: self.IcName = IcName


    def CalculateDistFromRightMostPointToArbPoint(self, Point: point):
        return Point.CalculateDistanceToOtherPoint(point(self.DownRightMostPoint.x, self.UpLeftMostPoint.y))

    def CalculateDistFromleftMostPointToArbPoint(self, Point):
        return Point.CalculateDistanceToOtherPoint(self.UpLeftMostPoint)

    def printName(self):
        print(self.IcName)