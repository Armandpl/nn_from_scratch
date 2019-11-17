import numpy as np
import matplotlib.image as mpimg
import os

def reluDerivative(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


def relu(x):
    x = np.maximum(x, 0)
    return x


def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

def load_mnist(directory_, inputs, outputs):
    for i in range(9):  # for each digit
        print("loading images of ", i)
        directory = directory_ + str(i) + "/"

        for filename in os.listdir(directory):
            if filename.endswith(".png"):
                img = mpimg.imread(directory + filename)
                inputs.append(np.reshape(img, (784, 1)))
                out = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                out[i] = 1
                outputs.append(np.transpose(np.matrix(out)))
                continue
            else:
                continue
