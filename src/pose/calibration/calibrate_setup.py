from src.pose.calibration.calib import Calib
from src.pose.utils.math_utils import dist

class CalibrateSetup:
    '''DOC'''
    
    def __init__(self):
        self.__done = False

    def is_done(self):
        '''DOC'''

        for calib in self.__calibrate_list:
            if not calib.is_stable():
                return False
        if self.__done is False:
            print("Calibrating done")
        self.__done = True
        return True
    
    def get_calibrations_value(self):
        '''DOC'''

        return [data.val() for data in self.__calibrate_list]
    
    def setup(self, fingers: list):
        '''DOC'''

        print("Calibration...")
        self.__calibrate_list = [Calib(finger) for finger in fingers]

    def evaluate(self, fingers, ground):
        '''DOC'''
        
        for i in range(len(fingers)):
            calculation = dist(fingers[i].get_joint_at(0), fingers[i].get_joint_at(-1)) / dist(ground[0], ground[1])
            self.__calibrate_list[i].add_data(calc=calculation)
            fingers[i].set_buffer(self.__calibrate_list[i].val())
            
    