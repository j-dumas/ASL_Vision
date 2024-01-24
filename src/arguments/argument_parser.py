import getopt, sys, time

class ArgumentParser:
    ''' The Argument Parser parses arguments coming from the command-line as booleans and integers. '''

    def __init__(self) -> None:
        self.__help, self.__debug, self.__train, self.__restart = False, False, False, False
        self.__epochs, self.__nb_of_letters = None, None
        self.__short_opt = "de:hrt"
        self.__long_opt = ["debug", "epochs=", "help", "train", "restart", "nb_of_letters="]

    def is_help_active(self) -> bool:
        return self.__help

    def is_training_active(self) -> bool:
        return self.__train

    def is_debug_active(self) -> bool:
        return self.__debug

    def is_restart_active(self) -> bool:
        return self.__restart

    def get_epochs(self) -> int:
        return self.__epochs

    def get_nb_of_letters(self) -> int:
        return self.__nb_of_letters

    def parse_arguments(self) -> None:
        '''
        List of arguments available:
            - e or epochs:
                Number of times the AI will pass over the training set.
                Takes a value.

            - h or help:
                Classic — Prints out the help

            - t or train:
                Tells the AI to train instead of predict.

            - r or restart:
                Tell the AI to restart itself by regenerating weights and biases
                instead of reading the data file.

            - nb_of_letters:
                Number of letter instances for each letter contained in the
                training set. Takes a value.
        '''

        full_cmd_arguments = sys.argv
        argument_list = full_cmd_arguments[1:]
        arguments = self.__get_parameters(argument_list)
        for current_argument, current_value in arguments:
            self.__check_help(current_argument)
            if self.__help:
                return
            self.__check_training(current_argument)
            self.__check_debug(current_argument)
            self.__check_epochs(current_argument, current_value)
            self.__check_restart(current_argument)
            self.__check_nb_of_letters(current_argument, current_value)

    def __get_parameters(self, argument_list) -> None:
        try:
            arguments, values = getopt.getopt(argument_list, self.__short_opt, self.__long_opt)
        except getopt.error as err:
            print(str(err))
            sys.exit(2)
        return arguments

    def __check_help(self, current_argument) -> None:
        if current_argument in ("-h", "--help"):
            self.__help = True

    def __check_training(self, current_argument) -> None:
        if current_argument in ("-t", "--train"):
            print("Starting training")
            self.__train = True
            time.sleep(2)
    
    def __check_debug(self, current_argument) -> None:
        if current_argument in ("-d", "--debug"):
            if not self.__train:
                print("Debug activated")
                self.__debug = True

    def __check_epochs(self, current_argument, current_value) -> None:
        if current_argument in ("-e", "--epochs"):
            if self.__train:
                print(f"Number of epochs: {current_value}")
                self.__epochs = current_value
            else:
                print("You need to add -t or --train to add epochs to the training")
                print("Ignoring epochs...")
            time.sleep(1)

    def __check_restart(self, current_argument) -> None:
        if current_argument in ("-r", "--restart"):
            if self.__train:
                print("Restarting training")
                self.__restart = True
                time.sleep(1)
            else:
                print("You need to add -t or --train to train the model after restarting it")
                print("Exiting...")
                sys.exit(2)

    def __check_nb_of_letters(self, current_argument, current_value) -> None:
        if current_argument in ("--nb_of_letters"):
            if self.__train:
                print(f"Number of instances per letters: {current_value}")
                self.__nb_of_letters = current_value
            else:
                print("You need to add -t or --train to add instances of letters to the training")
                print("Ignoring...")
            time.sleep(1)