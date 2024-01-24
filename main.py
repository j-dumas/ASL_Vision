from src.ai.nn_builder import NeuralNetworkBuilder
from src.ai.neural_network import NeuralNetwork
from src.ai.training import AiTraining
from src.arguments.argument_parser import ArgumentParser
from src.pose.cam import Cam

import numpy as np
np.set_printoptions(suppress=True)

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.parse_arguments()

    builder = NeuralNetworkBuilder()

    if parser.is_help_active():
        with open('doc/help.txt', 'r') as helpFile:
            print(helpFile.read())
        exit()

    elif parser.is_training_active():
        nb_per_set = parser.get_nb_of_letters()
        if nb_per_set is not None:
            aiTraining = AiTraining(nb_per_set=int(nb_per_set))
        else:
            aiTraining = AiTraining()
        aiTraining.generate()
        training_set = aiTraining.get_training_set()
        training_labels = aiTraining.get_training_labels()

        epochs = parser.get_epochs()
        if epochs is not None:
            builder.with_epochs(int(epochs))

        restart = parser.is_restart_active()
        if not restart:
            builder.with_saved_data()

        nn = builder.build()

        nn.train(training_set, training_labels)
        nn.save_data()

    else:
        builder = NeuralNetworkBuilder()
        builder.with_saved_data()
        nn = builder.build()

        cam = Cam(neural_network=nn, debug=parser.is_debug_active())
        cam.read()