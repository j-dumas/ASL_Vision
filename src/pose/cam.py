import time
import cv2
import mediapipe as mp
import numpy as np
import string

from src.pose.utils.hand_utils import *
from src.pose.utils.list_utils import *
import src.pose.utils.utils as utils

from src.pose.finger import Finger as Finger
from src.pose.hand import *
from src.pose.calibration.calibrate_setup import CalibrateSetup
import src.pose.parts as parts

class Cam:
    '''DOC'''

    ALPHABET = string.ascii_lowercase

    def __init__(self, neural_network, debug=False):
        self.__debug = debug

        # CAMERA & SCREEN
        self.__cap = cv2.VideoCapture(0)
        self.__image = None
        self.__height = 0
        self.__width = 0
        self.__mpDraw = mp.solutions.drawing_utils

        # HANDS
        self.__mpHands = mp.solutions.hands
        self.__hands = self.__mpHands.Hands()
        self.__handsResults = None
        self.__handsLandmarks = None

        self.__handIndex = -1 
        self.__handL = Hand(HandType.LEFT)
        self.__handR = Hand(HandType.RIGHT)

        # BODY
        self.__mpPose = mp.solutions.pose
        self.__pose = self.__mpPose.Pose()
        self.__bodyResults = None
        self.__bodyLandmarks = None

        self.__setup = CalibrateSetup()
        self.__setup.setup(self.__handL.get_fingers())

        # NEURAL NETWORK
        self.__nn = neural_network
        self.__letter = ""

    def read(self):
        '''DOC'''
        self.__start = time.time()
        while self.__cap.isOpened():
            success, self.__image = self.__cap.read()
            imgRGB = cv2.cvtColor(self.__image, cv2.COLOR_BGR2RGB)

            self.__handsResults = self.__hands.process(imgRGB)
            self.__bodyResults = self.__pose.process(imgRGB)
            self.__height, self.__width, c = self.__image.shape

            if self.__handsResults.multi_hand_landmarks and self.__bodyResults.pose_landmarks:
                self.__handsLandmarks = self.__handsResults.multi_hand_landmarks
                nbHands = len(self.__handsResults.multi_hand_landmarks)

                # Boucle des deux mains (hands).
                for i in range(nbHands):
                    self.__handIndex = i
                    if self.__debug:
                        self.__mpDraw.draw_landmarks(self.__image, self.__handsLandmarks[i], self.__mpHands.HAND_CONNECTIONS)

                    wrist = self.__handsLandmarks[i].landmark[parts.HandParts.WRIST]
                    Hx, Hy = int(wrist.x * self.__width), int(wrist.y * self.__height)
                    lowerDistance = -1
                    
                    # Boucle des parties du corps (poses).
                    for bodyPart, self.__bodyLandmarks in enumerate(self.__bodyResults.pose_landmarks.landmark):
                        if (bodyPart == parts.BodyParts.INDEX_L or bodyPart == parts.BodyParts.INDEX_R):
                            BPx, BPy = int(self.__bodyLandmarks.x * self.__width), int(self.__bodyLandmarks.y * self.__height)

                            if nbHands == 1:
                                self.__reset_hands_index()
                                self.__set_hands_info_according_to_hand_part(bodyPart, Hx, Hy, i)
                                    
                            elif nbHands == 2:
                                distance = self.__get_distance_between_hand_parts_and_body_parts(BPx, BPy, Hx, Hy)
                                
                                if distance < lowerDistance or lowerDistance == -1:
                                    lowerDistance = distance

                                    if (bodyPart == parts.BodyParts.INDEX_R):
                                        self.__handR.set_index(i)
                                        if (i == 0): self.__handL.set_index(1)
                                        else: self.__handL.set_index(0)

                                self.__set_hands_info_according_to_hand_index(Hx, Hy, i)

                    cv2.circle(imgRGB, (Hx, Hy), 12, (0,0,255), cv2.FILLED)
                    self.__send_to_nn()
                                        
            else:
                self.__reset_hands_informations()

            self.__display_hand_informations()

            cv2.imshow("Render", self.__image)
            cv2.waitKey(1)

    def __display_text(self, image, text, index, size=1, color=(255, 0, 0)):
        cv2.putText(image, text, (1, index*20), cv2.FONT_HERSHEY_COMPLEX_SMALL, size, color, size)

    def __display_hand_informations(self):
        if self.__setup.is_done():
            if self.__debug:
                self.__display_text(self.__image, "L POSITION:" + str(self.__handL.get_x()), 1)
                self.__display_text(self.__image, "L ORIENTATION:" + str(self.__handL.get_orientation()), 2)
                self.__display_text(self.__image, "L ROTATION:" + str(self.__handL.get_rotation()), 3)
                self.__display_text(self.__image, "L PINKY:" + str(self.__handL.get_finger(0).get_stage()), 4)
                self.__display_text(self.__image, "L RING:" + str(self.__handL.get_finger(1).get_stage()), 5)
                self.__display_text(self.__image, "L MIDDLE:" + str(self.__handL.get_finger(2).get_stage()), 6)
                self.__display_text(self.__image, "L INDEX:" + str(self.__handL.get_finger(3).get_stage()), 7)
                self.__display_text(self.__image, "L THUMB:" + str(self.__handL.get_finger(4).get_stage()), 8)
                self.__display_text(self.__image, "R POSITION:"+ str(self.__handR.get_x()), 9)
                self.__display_text(self.__image, "R ORIENTATION:" + str(self.__handR.get_orientation()), 10)
                self.__display_text(self.__image, "R ROTATION:" + str(self.__handR.get_rotation()), 11)
                self.__display_text(self.__image, "R PINKY:" + str(self.__handR.get_finger(0).get_stage()), 12)
                self.__display_text(self.__image, "R RING:" + str(self.__handR.get_finger(1).get_stage()), 13)
                self.__display_text(self.__image, "R MIDDLE:" + str(self.__handR.get_finger(2).get_stage()), 14)
                self.__display_text(self.__image, "R INDEX:" + str(self.__handR.get_finger(3).get_stage()), 15)
                self.__display_text(self.__image, "R THUMB:" + str(self.__handR.get_finger(4).get_stage()), 16)
            self.__display_text(self.__image, f'Letter is {self.__letter.upper()}', 22, size=2, color=(0, 0, 255))
        else:
            self.__display_text(self.__image, "Calibrate", 17)

    def __is_in_screen(self, x, y):
        if (x*self.__width < 0 or x*self.__width > self.__width): return False
        if (y*self.__height < 0 or y*self.__height > self.__height): return False
        return True

    def __get_distance_between_hand_parts_and_body_parts(self, BPx, BPy, Hx, Hy):
        return utils.distance_between(BPx, BPy, Hx, Hy)

    def __reset_hands_index(self):
        self.__handL.set_index(-1)
        self.__handR.set_index(-1)
        
    def __reset_hands_informations(self):
        self.__handL.reset()
        self.__handR.reset()

    def __set_hands_info_according_to_hand_part(self, bodyPart, Hx, Hy, i):
        if self.__is_in_screen(self.__bodyLandmarks.x, self.__bodyLandmarks.y):
            hand = self.__handL
            if (bodyPart == parts.BodyParts.INDEX_L): self.__handR.reset()
            if (bodyPart == parts.BodyParts.INDEX_R):
                self.__handL.reset()
                hand = self.__handR

            hand.set_position(Hx,Hy)
            self.__set_hands_rotation(hand, i)
            self.__set_hand_orientation(hand, i)
            self.__set_fingers_infos(hand, i)

    def __set_hands_info_according_to_hand_index(self, hx, hy, i):
        if (self.__handIndex == 0):
            if (self.__handL.get_index() == 0):
                self.__set_hand_all_infos(self.__handL, hx, hy, i)
            elif (self.__handR.get_index() == 0):
                self.__set_hand_all_infos(self.__handR, hx, hy, i)
        elif (self.__handIndex == 1):
            if (self.__handL.get_index() == 1):
                self.__set_hand_all_infos(self.__handL, hx, hy, i)
            elif (self.__handR.get_index() == 1):
                self.__set_hand_all_infos(self.__handR, hx, hy, i)

    def __set_hand_all_infos(self, hand, hx, hy, i):
        hand.set_position(hx, hy)
        self.__set_hands_rotation(hand, i)
        self.__set_hand_orientation(hand, i)
        self.__set_fingers_infos(hand, i)

    def __set_fingers_infos(self, hand, i):
        update_fingers(self.__handsLandmarks[i].landmark, fingers=hand.get_fingers())
        ground = [hand.get_finger(3).get_last_joint(), hand.get_finger(0).get_last_joint()]

        self.__setup.evaluate(hand.get_fingers(), ground)

        for finger in hand.get_fingers():
            finger.set_stage(get_finger_stage(finger, ground, is_thumb = finger.is_thumb(), is_pinky= finger.is_pinky(), is_right_hand=(hand.get_type() != 1), hand=hand))

    def __set_hand_orientation(self, hand, i):
        thumb = self.__handsLandmarks[i].landmark[parts.HandParts.THUMB_T]
        pinky = self.__handsLandmarks[i].landmark[parts.HandParts.PINKY_T]
        
        if (hand.get_type() == HandType.LEFT and thumb.x < pinky.x) or (hand.get_type() == HandType.RIGHT and thumb.x > pinky.x):
            if (hand.get_rotation() == HandRotation.UP): hand.set_orientation(HandOrientation.FRONT)
            if (hand.get_rotation() == HandRotation.DOWN): hand.set_orientation(HandOrientation.BACK)
        else: hand.set_orientation(HandOrientation.BACK)

    def __set_hands_rotation(self, hand, i):
        wirst = self.__handsLandmarks[i].landmark[parts.HandParts.WRIST]
        handLMs = self.__handsLandmarks[i].landmark
        fingers = [handLMs[parts.HandParts.THUMB_T], handLMs[parts.HandParts.INDEX_T], handLMs[parts.HandParts.MIDDLE_T], handLMs[parts.HandParts.RING_T] ,handLMs[parts.HandParts.PINKY_T]]

        counter = 0
        for j, finger in enumerate(fingers):
            xDistance, yDistance = utils.positive(finger.x - wirst.x), utils.positive(finger.y - wirst.y)
            
            if yDistance > xDistance:
                if (finger.y > wirst.y):
                    j, counter = self.__set_hand_rotation(j, counter, hand, HandRotation.DOWN)
                elif (finger.y < wirst.y):
                    j, counter = self.__set_hand_rotation(j, counter, hand, HandRotation.UP)
            else:
                if (hand.get_type() == HandType.LEFT and finger.x > wirst.x) or (hand.get_type() == HandType.RIGHT and finger.x < wirst.x):
                    j, counter = self.__set_hand_rotation(j, counter, hand, HandRotation.LEFT)
                if (hand.get_type() == HandType.LEFT and finger.x < wirst.x) or (hand.get_type() == HandType.RIGHT and finger.x > wirst.x):
                    j, counter = self.__set_hand_rotation(j, counter, hand, HandRotation.RIGHT)


    def __set_hand_rotation(self, j, counter, hand, handRotation):
        counter += 1
        if (j == 4):
            if counter >= 4: hand.set_rotation(handRotation)
            else: j = counter = 0
        return j, counter

    def __build_array(self):
        data = []
        data.append(self.__handR.get_finger(4).get_stage())
        data.append(self.__handR.get_finger(3).get_stage())
        data.append(self.__handR.get_finger(2).get_stage())
        data.append(self.__handR.get_finger(1).get_stage())
        data.append(self.__handR.get_finger(0).get_stage())
        data.append(self.__handR.get_orientation())
        data.append(self.__handR.get_rotation())
        return data

    def __send_to_nn(self):
        if (time.time() - self.__start) >= 2 and self.__setup.is_done():
            self.__start = time.time()
            data = self.__build_array()
            result = self.__nn.predict(np.array(data))
            max_index = np.argmax(result)
            self.__letter = self.ALPHABET[max_index]