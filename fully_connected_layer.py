import numpy as np
from layer import Layer
from utils import relu

class FullyConnectedLayer(Layer):

    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.layers = []

        # we initalize the weights and bias at random
        self.bias = np.random.random((self.output_size, 1))
        self.W = np.random.random((self.output_size, self.input_size))

    # let's say we use ReLU all the time
    # we also need to keep trace of the value before activation
    # to compute the gradient later
    def forward(self, input_):
        # on return un tuple avec l'output avant et après activation
        output = self.W.dot(input_) + self.bias
        return (output, relu(output))

    #returns previous error + return gradient
    def backprop(self, error, previous_z):
        # compute the gradient
        return relu(previous_z).dot(np.transpose(error))
