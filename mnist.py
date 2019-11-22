import numpy as np
from utils import relu
from utils import load_mnist
from neural_net import NeuralNet
from fully_connected_layer import FullyConnectedLayer
from utils import shuffle_in_unison
from utils import sigmoid

inputs = []
outputs = []

i_test = []
o_test = []

#Loading MNIST
print("Loading training set")
load_mnist("mnist_png/training/", inputs, outputs)

print("Loading test set")
load_mnist("mnist_png/testing/", i_test, o_test)

nn = NeuralNet()
nn.addLayer(FullyConnectedLayer(784, 16, activation=relu))
nn.addLayer(FullyConnectedLayer(16, 16, activation=relu))
nn.addLayer(FullyConnectedLayer(16, 10, activation=relu))

#no 1_cycle, 0.01, >10 epochs ?

nn.fit(inputs, outputs, 0.05, 20, batch_size=256, inputs_test=i_test, outputs_test=o_test)
