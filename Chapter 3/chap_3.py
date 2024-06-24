import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# dense layer
class Layer_Dense:

    # layer initialization
    def __init__(self, n_inputs, n_neurons):
        # initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # forward pass
    def forward(self, inputs):
        # calculate output values from inputs, weights, and biases 
        self.output = np.dot(inputs, self.weights) + self.biases

# create dataset
X, y = spiral_data(samples=100, classes=3)

# create dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2,3)

# perform forward pass of training data through this layer
dense1.forward(X)

# see output of first few samples
print(dense1.output[:5])