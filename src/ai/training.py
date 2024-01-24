import json

import string

alphabet = string.ascii_lowercase

import random
from datetime import datetime

import numpy as np
np.set_printoptions(suppress=True)

class AiTraining:
    '''Class that generates a training set based on the sign_language_dictionary.json file'''

    __VALUE_DIFFERENCE = 0.02
    __MIN_FINGER_VALUE = 0
    __MAX_FINGER_VALUE = 1

    def __init__(self, nb_per_set=10) -> None:
        self.__json_dict = dict()
        self.__training_set = []
        self.__training_labels = []
        self.__nb_per_set = nb_per_set
        self.__read_json_dict()

    def __read_json_dict(self) -> None:
        with open('doc/sign_language_dictionary.json', 'r', encoding='utf-8') as f:
            self.__json_dict = json.load(f)

    def generate(self) -> None:
        '''
        For each letters, it generates a a given number of random values close to the value
        the finger should be to form the letter. The number of instances of letter generated
        is given by the nb_per_set in the constructor.

        It also generates labels, an array telling if the letter generated in the training set
        is the letter of the index of the label array.
        '''
        for letter in alphabet:
            self.__generate_set(letter)

    def __generate_set(self, letter) -> None:
        random.seed(datetime.now())
        perfect_result = self.__json_dict[letter]['fingers']
        for i in range(self.__nb_per_set):
            data = []
            for i in range(5):
                finger = perfect_result[i]
                finger_max = finger if finger >= self.__MAX_FINGER_VALUE else min(self.__MAX_FINGER_VALUE, finger + self.__VALUE_DIFFERENCE)
                finger_min = finger if finger <= self.__MIN_FINGER_VALUE else max(self.__MIN_FINGER_VALUE, finger - self.__VALUE_DIFFERENCE)
                random_finger_value = random.uniform(finger_min, finger_max)
                data.append(round(random_finger_value, 4))
            data.append(self.__json_dict[letter]['orientation'])
            data.append(self.__json_dict[letter]['rotation'])
            self.__training_set.append(data)
            self.__generate_label(letter)

    def __generate_label(self, letter) -> None:
        label = [0] * 26
        index = alphabet.index(letter)
        label[index] = 1
        self.__training_labels.append(label)

    def get_training_set(self, numpy_array=True) -> np.array:
        return np.array(self.__training_set) if numpy_array else self.__training_set

    def get_training_labels(self, numpy_array=True) -> np.array:
        return np.array(self.__training_labels) if numpy_array else self.__training_labels

