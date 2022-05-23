
import math

class point:
    x = 0
    y = 0
    
    ConnectedToChip = False

    def __init__(self, x = None, y = None, ConnectedToChip = None):
        if x: self.x = x
        else: self.x = 0
        
        if y: self.y = y
        else: self.y = 0

        # TODO: have some kind of check to make sure the point is actually close to the IC
        if ConnectedToChip: self.ConnectedToChip = ConnectedToChip

    def CalculateDistanceToOtherPoint(self, otherPoint):
        return math.sqrt( (self.x - otherPoint.x) ** 2 + (self.y - otherPoint.y) ** 2)