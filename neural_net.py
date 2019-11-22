import numpy as np
from utils import reluDerivative
from utils import relu
from utils import sigmoid_derivative
from utils import shuffle_in_unison
from layer import Layer
import progressbar

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

    def fit(self, inputs, outputs, max_lr, epochs, batch_size = None, inputs_test = None, outputs_test = None):
        lowest_lr = max_lr/5
        lr = lowest_lr
        increment = (max_lr-lowest_lr)/epochs/2
        # lr = max_lr

        for i in range(epochs):
            print("epoch: ",i+1,"/",epochs)
            inputs_shuffled = inputs
            outputs_shuffled = outputs
            mse = None

            # mini batch shuffle
            if batch_size == None:
                batch_nb = 1
            else:
                batch_nb = len(inputs)//batch_size
                ## shuffle in unison let us shuffle inputs and outputs in the same way
                ## this way outputs[0] still correspond to inputs[0] after shuffling
                shuffle_in_unison(inputs_shuffled, outputs_shuffled)

            if i < epochs/2:
                lr += increment
            else:
                lr -= increment

            for b in progressbar.progressbar(range(batch_nb)): # for each mini batch

                ## compute the gradient for the inputs
                final_grad_w = []
                final_grad_b = []

                ##
                if batch_size == None:
                    R = len(inputs_shuffled)
                else:
                    R = batch_size

                for j in range(R): ## for each input in the batch
                    ## we compute the gradient and do gradient descent

                    if batch_size == None:
                        idx = j
                    else:
                        idx = j+b*batch_size
                    # batch 0 : idx [0, batch_size]
                    # batch 1 : idx [batch_size, batch_size+batch_size]
                    # batch 2 : idx [batch_size*2, batch_size*3]
                    # ...

                    back = self.backprop(inputs_shuffled[idx], outputs_shuffled[idx])

                    # if it is the first input of the batch we can't average the gradient yet
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

            print("current weight per layer")
            for u in range(len(self.layers)):
                print("layer ",u)
                print(self.layers[u].W)

            print("MSE:", mse)

            # if a test set is provided we compute some metrics
            if inputs_test is None:
                print('') # todo: alternative to print newlines
            else:
                correct_answers = 0
                for t in range(len(inputs_test)):
                    fwd = self.forward(inputs_test[t])
                    y_pred = fwd[0]
                    y_true = outputs_test[t]

                    #y_pred[y_pred >= 0.5] = 1
                    #y_pred[y_pred < 0.5] = 0

                    #if np.array_equal(y_pred, y_true):
                    #    correct_answers += 1
                    if np.argmax(y_pred) == np.argmax(y_true):
                        correct_answers += 1

                accuracy = (correct_answers/len(inputs_test))*100

                print("correct answers: ",correct_answers,"/",len(inputs_test))
                print("Accuracy: ",accuracy,"%")
            print("")

    # return full gradient for one input
    def backprop(self, input_, target):
        w_grad = []
        b_grad = []

        # feed forward
        out = self.forward(input_)
        y_pred = out[0]
        Z = out[1]
        squared_error = np.mean ( np.power ( np.subtract( y_pred, target ), 2) )

        # hardcoded error + activation
        # don't forget it's a Hadamard product
        # we start w/ the output error

        # todo : get the right derivative from layer properties

        curr_err = np.multiply( np.subtract( y_pred, target ), reluDerivative(Z[-1]))

        for i in reversed(range(len(self.layers))):

            if i == 0:
                previous_a = input_
            else:
                previous_a = self.layers[i-1].activation(Z[i-1])

            backprop_res=self.layers[i].backprop(curr_err, previous_a)

            #we append the gradient of this layer to our gradient matrix
            w_grad.append(backprop_res)
            b_grad.append(curr_err)

            # compute the current error
            WT = np.transpose(self.layers[i].W)
            curr_err  = np.multiply( WT.dot( curr_err ), previous_a )

        # element in the grad list were added backward, let's reverse them
        w_grad.reverse()
        b_grad.reverse()

        return (w_grad, b_grad, squared_error)
