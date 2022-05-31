

from chip import *
from point import *

class PrintedCircutBoard:

    # Dimensions = x,y ?
    # layers = [] ?
    EntireBoardPoints = []
    chips = []

    def __init__(self, EntireBoardPoints: list[point] = None, chips: list[chip] = None ) -> None:
        
        if EntireBoardPoints:
            self.EntireBoardPoints = EntireBoardPoints
            for point in self.EntireBoardPoints:
                point.ConnectedToPCB = self

        if chips: 
            self.chips = chips
            for chip in self.chips:
                chip.ConnectedToPCB = self

    def ReturnAllNCpoints(self) -> list[point]:
        return [point for point in self.EntireBoardPoints if not point.ConnectedToChip]

    def addChip(self, Chip: chip) -> None:
        self.chips.append(Chip)
        #print(f"In add chip! ic name: {Chip.IcName}")
        Chip.ConnectedToPCB = self
        for Point in Chip.pins:
            self.addPoint(Point)

    def addPoint(self, Point: point) -> None:
        # Do note that we dont check proximty here, could pose a problem
        if Point not in self.EntireBoardPoints:
            self.EntireBoardPoints.append(Point)
            Point.ConnectedToPCB = self
    
    def printInfo(self, Verbose = 0) -> None:
        if Verbose > 0:
            for chip in self.chips:
                chip.printInfo()
        if Verbose > 1:
            for EBP in self.EntireBoardPoints:
                print(f"Point: {EBP.x}, {EBP.y}")
    
