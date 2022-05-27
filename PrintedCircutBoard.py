

from chip import *
from point import *

class PrintedCircutBoard:

    # Dimensions = x,y ?
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

    def addChip(self, chip: chip) -> None:
        self.chips.append(chip)
        chip.ConnectedToPCB = self
        for point in chip.pins:
            self.addPoint(point)

    def addPoint(self, point: point) -> None:
        # Do note that we dont check proximty here, could pose a problem
        if point not in self.EntireBoardPoints:
            self.EntireBoardPoints.append(point)
            point.ConnectedToPCB = self
    
