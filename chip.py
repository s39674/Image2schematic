import point

class chip:
    
    name = "blank"
    pins = []
    point1 = point()
    point2 = point()
    
    def __init__(self, point1, point2, name, pins: list(point)):
        self.name = name
        self.pins = pins
        self.point1 = point1
        self.point2 = point2

    def __init__(self, name):
        self.name = name
        self.pins = []
        self.point1 = point()
        self.point2 = point()
    
    def __init__(self):
        self.name = "blank"
        self.pins = []
        self.point1 = point()
        self.point2 = point()

    def CalculateDistFromRightMostPoint(self):
        pass

    def CalculateDistFromleftMostPoint(self):
        pass