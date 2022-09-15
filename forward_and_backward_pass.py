import numpy as np

# Passed in gradients from next layer
# Incremental values considered for example
dvalues = np.array([[1., 1., 1.],
                    [2., 2., 2.],
                    [3., 3., 3.]])

# 3 sets of input samples
inputs = np.array([[1, 2, 3, 2.5],
                    [2, 5, -1, 2],
                    [-1.5, 2.7, 3.3, -0.8]])

# 4 inputs thus 4 weights
# Need to keep the weights transposed
weights = np.array([[0.2, 0.8, -0.5, 1],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]]).T

# One bias for each row
biases = np.array([[2, 3, 0.5]])

# Forward pass
layer_outputs = np.dot(inputs, weights) + biases
relu_outputs = np.maximum(0, layer_outputs)

# Testing backpropagation
# ReLU activation - stimulates derivative wrt to input values from next layer passed to current layer during backpropagation
drelu = relu_outputs.copy()
drelu[layer_outputs <= 0] = 0

# Dense Layer 
# - 
# - 

# dinputs - multiplied by weights
dinputs = np.dot(drelu, weights.T)

# dweights - multiply by inputs
dweights = np.dot(inputs.T, drelu)

dbiases = np.sum(drelu, axis=0, keepdims=True)

# Updating parameters 
weights += -0.001 * dweights
biases += -0.001 * dvfbiases

print(weights)
print(biases)




