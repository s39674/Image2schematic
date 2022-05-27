
import math

class point:
    x = 0
    y = 0
    
    ConnectedToChip = None
    ConnectedToPCB = None

    def __init__(self, x = None, y = None, ConnectedToChip = None, ConnectedToPCB = None):
        if x: self.x = x
        else: self.x = 0
        
        if y: self.y = y
        else: self.y = 0

        # TODO: have some kind of check to make sure the point is actually close to the IC
        if ConnectedToChip: self.ConnectedToChip = ConnectedToChip

        if ConnectedToPCB: self.ConnectedToPCB = ConnectedToPCB

    def CalculateDistanceToOtherPoint(self, otherPoint):
        return math.sqrt( (self.x - otherPoint.x) ** 2 + (self.y - otherPoint.y) ** 2)

    def IsCloseToOtherPoint(self, otherPoint, rel_tol: float = 0.15, abs_tol: float = 10.0) -> bool:
        
        if(math.isclose(self.x, otherPoint.x, rel_tol=rel_tol, abs_tol=abs_tol)):
        # x is close enough
        # check for y; ~10 pixels abs margin of error
            if(math.isclose(self.y, self.y, rel_tol=rel_tol, abs_tol=abs_tol)):
                return True

        return False