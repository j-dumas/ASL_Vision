from abc import ABC, abstractmethod

from src.ai.neural_network import NeuralNetwork

class Builder(ABC):

    @abstractmethod
    def with_saved_data(self):
        pass

    @abstractmethod
    def with_epochs(self, nb_of_epochs):
        pass

    @abstractmethod
    def with_learning_rate(self, learning_rate):
        pass

    @abstractmethod
    def with_nb_inputs(self, nb_inputs):
        pass

    @abstractmethod
    def build(self):
        pass

class NeuralNetworkBuilder(Builder):

    def __init__(self) -> None:
        self.__nn = NeuralNetwork()
        self.__used_data = False
    
    def with_saved_data(self) -> None:
        '''Tells the Neural Network to use its saved data'''
        self.__nn.use_saved_data()
        self.__used_data = True

    def with_epochs(self, nb_of_epochs) -> None:
        '''Sets the number of epochs'''
        self.__nn.change_epochs(nb_of_epochs)

    def with_learning_rate(self, learning_rate) -> None:
        '''Sets the learning rate'''
        self.__nn.change_learning_rate(learning_rate)

    def with_nb_inputs(self, nb_inputs) -> None:
        '''Sets the number of inputs'''
        self.__nn.change_nb_inputs(nb_inputs)

    def build(self) -> NeuralNetwork:
        '''Builds the Neural Network Object'''
        if not self.__used_data:
            self.__nn.use_new_data()
        
        return self.__nn