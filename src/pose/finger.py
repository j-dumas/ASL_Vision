class Finger:
    '''DOC'''
    
    def __init__(self, ids, name=None, is_thumb = False, is_pinky = False):
        self.__ids = ids
        self.__name = name
        self.__finger_joints = dict()
        self.__stage = 0
        self.__is_thumb = is_thumb
        self.__is_pinky = is_pinky
        self.__buffer = 0
        self.__build()
        
    # Getters
    def is_thumb(self):
        return self.__is_thumb

    def is_pinky(self):
        return self.__is_pinky

    def get_all_joints(self):
        '''DOC'''
        return self.__finger_joints
    
    def get_joint_at(self, index):
        '''DOC'''
        return self.__finger_joints.get(self.__ids[index])
    
    def get_last_joint(self):
        '''DOC'''
        return self.get_joint_at(-1)
    
    def get_specific_joint(self, id):
        '''DOC'''
        return self.__finger_joints.get(id)
    
    def get_finger_name(self):
        '''DOC'''
        return ['Not-Defined-Yet', self.__name][self.__name != None]

    def get_stage(self):
        '''DOC'''
        return self.__stage

    def set_stage(self, value):
        '''DOC'''
        self.__stage = value
        
    def set_buffer(self, buffer):
        '''DOC'''
        self.__buffer = buffer

    def get_buffer(self):
        '''DOC'''
        return self.__buffer
    
    # Others
    def update(self, joints):
        '''DOC'''
        for id in self.__ids:
            x_pos = joints[id].x
            y_pos = joints[id].y
            self.__finger_joints.update({ id: [ x_pos, y_pos ]})

    def __build(self):
        print(f'Building finger({self.get_finger_name()})')
        for id in self.__ids:
            self.__finger_joints.update({ id: [ 0, 0 ] })
    
    def __str__(self):
        return f'The finger({self.get_finger_name()}) has {len(self.get_all_joints())} joints linked.'
    
    def __repr__(self):
        return "The Finger class is used to manage easily the finger's data from cv2."