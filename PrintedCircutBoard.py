

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
        else: self.EntireBoardPoints = [] # avoid "shared memory"

        if chips: 
            self.chips = chips
            for chip in self.chips:
                chip.ConnectedToPCB = self
        else: self.chips = [] # avoid "shared memory"

    def ReturnAllNCpoints(self) -> list[point]:
        return [point for point in self.EntireBoardPoints if not point.ConnectedToChip]

    def addChip(self, Chip: chip) -> None:
        self.chips.append(Chip)
        #print(f"In add chip! ic name: {Chip.IcName}")
        Chip.ConnectedToPCB = self
        for Point in Chip.pins:
            self.addPoint(Point)

    def addPoint(self, Point: point) -> None:
        # Do note that we dont check proximty here, could pose a duplication problem
        if Point not in self.EntireBoardPoints:
            self.EntireBoardPoints.append(Point)
            Point.ConnectedToPCB = self

    def ReturnPointsSameAsRef(self, refPoint: point) -> point:

        return [Point for Point in self.EntireBoardPoints if Point.IsCloseToOtherPoint(refPoint)]
    
    def ReturnPointsThatAreLike(self, refPoints: list[point], rel_tol: float = 0.15, abs_tol: float = 10.0) -> list[point]:
        """
        This function returns points that are like the reference point.
        """
        temp = []
        for refPoint in refPoints:
            for Point in self.EntireBoardPoints:
                if refPoint.IsCloseToOtherPoint(Point, rel_tol=rel_tol, abs_tol=abs_tol):
                    temp.append(Point)
                    break
        return temp

    def printInfo(self, Verbose = 0) -> None:
        if Verbose > 0:
            for chip in self.chips:
                chip.printInfo()
        if Verbose > 1:
            for EBP in self.EntireBoardPoints:
                print(f"Point: {EBP.x}, {EBP.y}")
    
