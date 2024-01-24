import numpy as np
np.set_printoptions(suppress=True)

import json
import progressbar as pb

# https://stackabuse.com/creating-a-neural-network-from-scratch-in-python-multi-class-classification/

class NeuralNetwork:
    '''Main class for the AI. Unless it is restarted, it reads its weights and biases from data.json.'''

    __HIDDEN_LAYER1_NODE_NB = 30

    def __init__(self) -> None:
        self.__weights, self.__bias = dict(), dict()
        self.__epochs = 20000
        self.__lr = 0.1
        self.__nb_inputs = 7
    
    def use_saved_data(self) -> None:
        '''Uses previously saved data'''
        self.load_data()
    
    def use_new_data(self) -> None:
        '''Restarts from scratch'''
        self.__init_weights()
        self.__init_bias()

    def change_epochs(self, nb_of_epochs) -> None:
        '''Ajusts the number of epochs'''
        self.__epochs = nb_of_epochs

    def change_learning_rate(self, learning_rate) -> None:
        '''Changes the learning rate'''
        self.__lr = learning_rate

    def change_nb_inputs(self, nb_inputs) -> None:
        '''Changes the number of inputs'''
        self.__nb_inputs = nb_inputs

    def __init_weights(self) -> None:
        self.__weights['wh1'] = np.random.rand(self.__nb_inputs, self.__HIDDEN_LAYER1_NODE_NB)
        self.__weights['wo'] = np.random.rand(self.__HIDDEN_LAYER1_NODE_NB, 26)

    def __init_bias(self) -> None:
        self.__bias['bh1'] = np.random.rand(self.__HIDDEN_LAYER1_NODE_NB)
        self.__bias['bo'] = np.random.rand(26)

    def __softmax(self, A, dimension=1) -> float:
        expA = np.exp(A)
        return expA / expA.sum(axis=dimension-1, keepdims=True)

    def __sigmoid(self, x) -> float:
        return 1/(1+np.exp(-x))

    def __sigmoid_der(self, x) -> float:
        return self.__sigmoid(x)*(1-self.__sigmoid(x))

    def train(self, feature_set, labels) -> None:
        '''
        Training function for the AI.
        Consists of two parts: feedforward and backpropagation.

        The feedforward first sends the inputs to the hidden layers' nodes by calculating
        the dot product of the inputs and the nodes' weights and adds the bias.
        It then passes the result into an activation function: the sigmoid function, 
        which squashes input values between 1 and 0.

        The AI would learn without the backpropagation, which is the function that updates
        the weights and biases for the AI to be more precise. First, it calculates the cost
        of the predictions by finding the difference between the predicted output and the 
        actual output. Then it updates the weights and biases by finding the minima. To find
        the minima, it uses a gradient decent that find the partial derivative of the cost 
        function with respect to each weight and bias and subtract the result from the existing 
        weight values to get the new weight values.

        For each epochs, the training function will run the feedforward and the backpropagation,
        updating the weights and biases to be more precise in the predictions.

        It is explained in depth here: https://stackabuse.com/creating-a-neural-network-from-scratch-in-python/
        '''
        bar = pb.ProgressBar(maxval=self.__epochs, \
            widgets=[pb.Bar('=', 'Training: [', ']'), \
            ' ', pb.Counter(), f'/{self.__epochs}'])
        bar.start()
        for epoch in range(self.__epochs):
            nodes_h1, activation_h1, activation_out = self.__feedforward(feature_set, epoch, labels)
            self.__back_propagation(feature_set, labels, nodes_h1, activation_h1, activation_out)
            bar.update(epoch+1)
        bar.finish()
    
    def __feedforward(self, feature_set, epoch, labels) -> np.array:
        nodes_h1 = np.dot(feature_set, self.__weights['wh1']) + self.__bias['bh1']
        activation_h1 = self.__sigmoid(nodes_h1)

        nodes_out = np.dot(activation_h1, self.__weights['wo']) + self.__bias['bo']
        activation_out = self.__softmax(nodes_out, dimension=2)
        if epoch % 1000 == 0:
                loss = np.sum(-labels * np.log(activation_out))
                print('Loss function value: ', round(loss, 6))
        return nodes_h1,activation_h1,activation_out

    def __back_propagation(self, feature_set, labels, nodes_h1, activation_h1, activation_out) -> None:
        dcost_dzo, dcost_wo, dcost_bo = self.__bp_phase1(labels, activation_h1, activation_out)
        dcost_wh, dcost_bh = self.__bp_phase2(feature_set, nodes_h1, dcost_dzo)
        self.__update(dcost_wo, dcost_bo, dcost_wh, dcost_bh)

    def __bp_phase1(self, labels, activation_h1, activation_out) -> float:
        dcost_dzo = activation_out - labels
        dzo_dwo = activation_h1

        dcost_wo = np.dot(dzo_dwo.T, dcost_dzo)

        dcost_bo = dcost_dzo
        return dcost_dzo,dcost_wo,dcost_bo

    def __bp_phase2(self, feature_set, nodes_h1, dcost_dzo) -> float:
        dzo_dah = self.__weights['wo']
        dcost_dah = np.dot(dcost_dzo , dzo_dah.T)
        dah_dzh = self.__sigmoid_der(nodes_h1)
        dzh_dwh = feature_set
        dcost_wh = np.dot(dzh_dwh.T, dah_dzh * dcost_dah)

        dcost_bh = dcost_dah * dah_dzh
        return dcost_wh,dcost_bh

    def __update(self, dcost_wo, dcost_bo, dcost_wh, dcost_bh) -> None:
        self.__weights['wh1'] -= self.__lr * dcost_wh
        self.__bias['bh1'] -= self.__lr * dcost_bh.sum(axis=0)

        self.__weights['wo'] -= self.__lr * dcost_wo
        self.__bias['bo'] -= self.__lr * dcost_bo.sum(axis=0)
    
    def predict(self, single_point) -> np.array:
        '''
        Predicts the single_point value between the choices using the dot product and
        the softmax activation function.
        '''
        pass1 = self.__sigmoid(np.dot(single_point, self.__weights['wh1']) + self.__bias['bh1'])
        return self.__softmax(np.dot(pass1, self.__weights['wo']) + self.__bias['bo'])

    def save_data(self) -> None:
        '''Writes the weights and biases into the data.json file'''
        json_weights = {}
        json_weights['wh1'] = self.__weights['wh1'].tolist()
        json_weights['wo'] = self.__weights['wo'].tolist()

        json_bias = {}
        json_bias['bh1'] = self.__bias['bh1'].tolist()
        json_bias['bo'] = self.__bias['bo'].tolist()

        data = {'weights': json_weights, 'bias': json_bias}
        
        with open('src/ai/data/data.json', 'w+', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def load_data(self) -> None:
        '''Reads the weights and biases from the data.json file'''
        with open('src/ai/data/data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)

            json_weights = data['weights']
            self.__weights['wh1'] = np.array(json_weights['wh1'])
            self.__weights['wo'] = np.array(json_weights['wo'])

            json_bias = data['bias']
            self.__bias['bh1'] = np.array(json_bias['bh1'])
            self.__bias['bo'] = np.array(json_bias['bo'])
