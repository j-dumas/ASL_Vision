from src.pose.utils.math_utils import avg, max

class Calib:
    '''DOC'''

    def __init__(self, finger: object, buffer: float = 32):
        self.finger = finger
        self.data = []
        self.buffer = buffer
        self.average = 0

    def val(self):
        '''DOC'''

        if self.is_stable():
            return self.average
        elif self.is_full():
            self.auto_clean()
        return -1

    def is_stable(self):
        '''DOC'''
        
        if not self.is_full():
            return False
        
        self.average = avg(self.data)
        calc = 0
        for i in range(len(self.data)):
            calc = max(calc, abs(self.average - self.data[i]))
        return calc <= 0.09
            
    def auto_clean(self):
        '''DOC'''

        if self.is_full():
            self.average = avg(self.data)
            self.data = self.data[-10:]
            
    def is_full(self):
        '''DOC'''

        return len(self.data) >= self.buffer
            
    def add_data(self, calc: float):
        '''DOC'''
        
        if not self.is_full():
            self.data.append(calc)