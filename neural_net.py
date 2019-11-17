import numpy as np
from utils import reluDerivative
from utils import relu
from layer import Layer

class NeuralNet:

    def __init__(self):
        self.layers = []

    def forward(self, input_):
        current = input_
        Z = []

        for i in range(len(self.layers)): # for each layer
            # [1] is to get the output after activation
            out = self.layers[i].forward(current)
            current = out[1]
            # we keep trace of the outputs before validation to compute the gradient
            Z.append(out[0])

        return (current, Z)

    def addLayer(self, layer):
        self.layers.append(layer)

    def fit(self, inputs, outputs, lr, epochs):
        for i in range(epochs):
            print("Epoch: ",i)
            final_grad_w = []
            final_grad_b = []
            mse = None

            ## compute the gradient for the inputs
            for j in range(len(inputs)):
                back = self.backprop(inputs[j], outputs[j])

                if j == 0:
                    final_grad_w = back[0]
                    final_grad_b = back[1]
                    mse = back[2]
                else:

                    for k in range(len(final_grad_w)):
                        # we average the nudges
                        final_grad_w[k] = np.true_divide(np.add(final_grad_w[k], back[0][k]), 2)
                        final_grad_b[k] = np.true_divide(np.add(final_grad_b[k], back[1][k]), 2)

                        mse = (mse+back[2])/2

            ## gradient descent
            for l in range(len(self.layers)):
                # for each layer we nudge the weights and biases
                self.layers[l].W = np.subtract (self.layers[l].W, np.transpose(final_grad_w[l])*lr)
                self.layers[l].bias = np.subtract ( self.layers[l].bias, final_grad_b[l]*lr)

            print("MSE:",mse)

    # return full gradient for one input
    # add mse ??
    def backprop(self, input_, target):
        w_grad = []
        b_grad = []

        # feed forward
        out = self.forward(input_)
        y_pred = out[0]
        Z = out[1]
        squared_error = np.power ( np.subtract( y_pred, target ), 2)

        # hardcoded error + activation
        # don't forget it's a Hadamard product
        # we start w/ the output error
        curr_err = np.multiply( np.subtract( y_pred, target ), reluDerivative(Z[len(Z)-1]))

        for i in reversed(range(len(self.layers))):

            if i == 0:
                previous_z = input_
            else:
                previous_z = Z[i-1]

            backprop_res=self.layers[i].backprop(curr_err, previous_z)

            #we append the gradient of this layer to our gradient matrix
            w_grad.append(backprop_res)
            b_grad.append(curr_err)

            # compute the current error
            WT = np.transpose(self.layers[i].W)
            curr_err  = np.multiply( WT.dot(curr_err), relu(Z[i-1]) )

        # element in the grad list were added backward, let's reverse them
        w_grad.reverse()
        b_grad.reverse()

        return (w_grad, b_grad, squared_error)
