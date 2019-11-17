import numpy as np
from utils import relu
from neural_net import NeuralNet
from fully_connected_layer import FullyConnectedLayer

#XOR Inputs
inputs = [np.matrix('0;0'), np.matrix('0;1'), np.matrix('1;0'), np.matrix('1;1')]

#XOR Outputs
outputs = np.array([[0],[1],[1],[0]])

nn = NeuralNet()
nn.addLayer(FullyConnectedLayer(2, 3))
nn.addLayer(FullyConnectedLayer(3, 1))

nn.fit(inputs, outputs, 0.05, 5000)

while True :
    data = input("enter data:")
    print(nn.forward(np.matrix(data))[0])
