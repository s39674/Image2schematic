
import chip

class point:

    x = 0
    y = 0
    
    def __init__(self):
        self.x = 0
        self.y = 0

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __init__(self, x, y, ConnectedToChip: chip):
        self.x = x
        self.y = y
        self.ConnectedToChip = ConnectedToChip