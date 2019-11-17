from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
from utils import relu
from utils import load_mnist
from neural_net import NeuralNet
from fully_connected_layer import FullyConnectedLayer
from utils import shuffle_in_unison

# load MNIST from server
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# training data : 60000 samples
# reshape and normalize input data
x_train = x_train.reshape(x_train.shape[0], 28*28, 1)
x_train = x_train.astype('float32')
x_train /= 255
# encode output which is a number in range [0,9] into a vector of size 10
# e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_train = np_utils.to_categorical(y_train)
y_jpp = []

for i in range(len(y_train)):
    y_jpp.append( np.transpose(np.matrix(y_train[i])) )

# same for test data : 10000 samples
x_test = x_test.reshape(x_test.shape[0], 28*28, 1)
x_test = x_test.astype('float32')
x_test /= 255
y_test = np_utils.to_categorical(y_test)

y_test_jpp = []

for i in range(len(y_test)):
    y_test_jpp.append(np.transpose(np.matrix(y_test[i])))

nn = NeuralNet()
nn.addLayer(FullyConnectedLayer(784, 16))
nn.addLayer(FullyConnectedLayer(16, 16))
nn.addLayer(FullyConnectedLayer(16, 10))

print('shape',nn.forward(x_train[5])[0].shape)

nn.fit(x_train, y_jpp, 0.0001, 3, batch_size=512,inputs_test=x_test, outputs_test=y_test_jpp)
