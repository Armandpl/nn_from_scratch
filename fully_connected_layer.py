import numpy as np
from layer import Layer
from utils import relu

class FullyConnectedLayer(Layer):

    def __init__(self, input_size, output_size, activation):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation

        # we initalize the weights and bias at random
        self.bias = np.random.random((self.output_size, 1))
        self.W = np.random.random((self.output_size, self.input_size))

    # we also need to keep trace of the value before activation
    # to compute the gradient later
    def forward(self, input_):
        # on return un tuple avec l'output avant et après activation
        output = self.W.dot(input_) + self.bias
        return (output, self.activation(output))

    #returns previous error + return gradient
    def backprop(self, error, previous_a):
        # compute the gradient
        return previous_a.dot(np.transpose(error))
