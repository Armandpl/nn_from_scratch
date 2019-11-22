import numpy as np
from layer import Layer
from utils import relu

class FullyConnectedLayer(Layer):

    def __init__(self, input_size, output_size, activation):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation

        # we initalize the weights at random and set the biases to be zeros
        self.bias = np.zeros((self.output_size, 1))

        # HE init : optimized for relu activation
        self.W = np.random.randn(self.output_size, self.input_size) * np.sqrt(2/self.input_size)

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
