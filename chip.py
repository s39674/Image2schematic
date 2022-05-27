
from point import *

class chip:
    
    IcName = None
    # an array to store all pointers to the pcb points that are connected to this ic
    pins = []
    # up left most cornor
    UpLeftMostPoint = point()
    # down right most cornor
    DownRightMostPoint = point()
    
    ConnectedToPCB = None

    def __init__(self, UpLeftMostPoint = None, DownRightMostPoint = None, IcName = None, pins: list[point] = None, ConnectedToPCB = None) -> None:
        if UpLeftMostPoint: self.UpLeftMostPoint = UpLeftMostPoint
        else: self.UpLeftMostPoint = point()

        if DownRightMostPoint: self.DownRightMostPoint = DownRightMostPoint
        else: self.DownRightMostPoint = point()

        if pins: 
            self.pins = pins
            # looping over recived pins to set the ConnectedToIC variable
            for point in pins: point.ConnectedToChip = self

        if IcName: self.IcName = IcName
        if ConnectedToPCB: self.ConnectedToPCB = ConnectedToPCB

    def CalculateDistFromRightMostPointToArbPoint(self, Point: point) -> float:
        return Point.CalculateDistanceToOtherPoint(point(self.DownRightMostPoint.x, self.UpLeftMostPoint.y))

    def CalculateDistFromleftMostPointToArbPoint(self, Point) -> float:
        return Point.CalculateDistanceToOtherPoint(self.UpLeftMostPoint)

    def returnName(self) -> str:
        return self.IcName