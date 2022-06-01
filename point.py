
import math

class point:
    x = 0
    y = 0
    
    ConnectedToChip = None
    ConnectedToPCB = None

    ConnectedToPoints = []

    # For future: hold the pin information from skidl search function
    # pinInfo = None
    
    def __init__(self, x: float = None, y: float = None, ConnectedToPoints = None, ConnectedToChip = None, ConnectedToPCB = None):
        if x: self.x = x
        else: self.x = 0
        
        if y: self.y = y
        else: self.y = 0

        # TODO: have some kind of check to make sure the point is actually close to the IC
        if ConnectedToChip: self.ConnectedToChip = ConnectedToChip
        if ConnectedToPoints: self.ConnectedToPoints = ConnectedToPoints
        # IF I DONT DO THIS it has "shared memory" with other points!
        else: self.ConnectedToPoints = []
        if ConnectedToPCB: self.ConnectedToPCB = ConnectedToPCB

    def ConnectToPoints(self, otherPoints = None) -> None:
        # Check if it's the current point or it is already in ConnectedToPoints 
        for Point in otherPoints:
            if Point not in self.ConnectedToPoints and not self.IsCloseToOtherPoint(Point):
                self.ConnectedToPoints.append(Point)
            

    def CalculateDistanceToOtherPoint(self, otherPoint):
        return math.sqrt( (self.x - otherPoint.x) ** 2 + (self.y - otherPoint.y) ** 2)

    def isEqual(self, refPoint):
        if self.x == refPoint.x and self.y == refPoint.y:
            return True
        return False

    def IsCloseToOtherPoint(self, otherPoint, rel_tol: float = 0.15, abs_tol: float = 10.0) -> bool:
        
        if(math.isclose(self.x, otherPoint.x, rel_tol=rel_tol, abs_tol=abs_tol)):
        # x is close enough
        # check for y; ~10 pixels abs margin of error
            if(math.isclose(self.y, otherPoint.y, rel_tol=rel_tol, abs_tol=abs_tol)):
                return True

        return False

    def printInfo(self, verbose: int = 0) -> str:
        if verbose > -1:
            return f"[{self.x}, {self.y}]"