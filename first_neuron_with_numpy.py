import numpy as np

# A single neuron
inputs = [1.0, 2.0, 3.0, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2.0

outputs = np.dot(inputs, weights) + bias

print(outputs)

# A layer of neurons
# Consider the above inputs for the layer

# Weights will be a 2D array since for every neuron a corresponding weight needs to be included
layer_weights = [[0.2, 0.8, -0.5, 1],
                 [0.5, -0.91, 0.26, -0.5],
                 [-0.26, -0.27, 0.17, 0.87]]
# Similarly there will be multiple biases\345
layer_biases = [2.0, 3.0, 0.5]

layer_outputs = np.dot(layer_weights, inputs) + layer_biases

print(layer_outputs)
