import numpy as np
from utils import relu
from utils import load_mnist
from neural_net import NeuralNet
from fully_connected_layer import FullyConnectedLayer
from utils import shuffle_in_unison

inputs = []
outputs = []

i_test = []
o_test = []

#Loading MNIST
print("Loading training set")
load_mnist("mnist_png/training/", inputs, outputs)

print("Loading test set")
load_mnist("mnist_png/testing/", i_test, o_test)

print("type ",type(inputs[0][5]))
print("shape", inputs[0][0].shape)

nn = NeuralNet()
nn.addLayer(FullyConnectedLayer(784, 16))
nn.addLayer(FullyConnectedLayer(16, 16))
nn.addLayer(FullyConnectedLayer(16, 10))

nn.fit(inputs, outputs, 0.0001, 3, batch_size=512, inputs_test=i_test, outputs_test=o_test)