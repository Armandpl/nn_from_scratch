class Layer:
    def __init__(self):
        print("dunno")

    def forward(self, input_):
        raise NotImplementedError

    def backprop(self, output, learning_rate):
        raise NotImplementedError