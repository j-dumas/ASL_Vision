from src.pose.finger import *
from src.pose.utils.hand_utils import get_finger_trail, update_fingers

class HandType():
    '''DOC'''

    LEFT = 0
    RIGHT = 1
    NONE = 2

class HandOrientation():
    '''DOC'''

    FRONT = 0
    SIDE = 1
    BACK = 2

class HandRotation():
    '''DOC'''

    UP = 0
    LEFT = 1
    DOWN = 2
    RIGHT = 3

class Hand():
    '''DOC'''

    def __init__(self, handType):
        self.__type = handType
        self.__index = -1
        self.__x = 0
        self.__y = 0
        self.__orientation = HandOrientation.FRONT
        self.__rotation = HandRotation.UP
        self.__fingers = [
            Finger(ids=get_finger_trail(20), name="pinky", is_pinky=True),
            Finger(ids=get_finger_trail(16), name="ring"),
            Finger(ids=get_finger_trail(12), name="middle"),
            Finger(ids=get_finger_trail(8), name="index"),
            Finger(ids=get_finger_trail(4), name="thumb", is_thumb=True)
            ]

    def set_type(self, type):
        '''DOC'''
        self.__type = type

    def get_type(self):
        '''DOC'''
        return self.__type

    def set_index(self, index):
        '''DOC'''
        self.__index = index

    def get_index(self):
        '''DOC'''
        return self.__index

    def set_position(self, x, y):
        '''DOC'''
        self.__x = x
        self.__y = y

    def get_x(self):
        '''DOC'''
        return self.__x

    def get_y(self):
        '''DOC'''
        return self.__y

    def set_rotation(self, rotation):
        '''DOC'''
        self.__rotation = rotation

    def get_rotation(self):
        '''DOC'''
        return self.__rotation

    def set_orientation(self, orientation):
        '''DOC'''
        self.__orientation = orientation

    def get_orientation(self):
        '''DOC'''
        return self.__orientation

    def set_fingers(self, listFinger):
        '''DOC'''
        self.__fingers = listFinger[::-1]

    def get_fingers(self):
        '''DOC'''
        return self.__fingers

    def get_finger(self, index):
        '''DOC'''
        if (index < len(self.__fingers)): return self.__fingers[index]

    def update_finger(self, landmarks):
        '''DOC'''
        update_fingers(landmarks, self.__fingers)

    def reset(self):
        '''DOC'''
        self.set_position(0, 0)
        self.set_orientation(HandOrientation.FRONT)
        self.set_rotation(HandRotation.UP)
        for finger in self.__fingers:
            finger.set_stage(0)