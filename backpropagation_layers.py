# Backpropagation in multiple layers
# In this code example only two layers are considered 

import numpy as np

# The passed in gradients from next layer is considered as 1 only for this example
dvalues = np.array([[1., 1., 1.]])

# As usual there will be 3 neurons in each layer but there are 4 inputs per neuron
# 3 neurons means 3 sets weights 
# 4 weights for 4 inputs
weights = np.array([[0.2, 0.8, -0.5, 1],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]]).T

dx0 = sum([dvalues[0][0]*weights[0][0], dvalues[0][1]*weights[0][1], dvalues[0][2]*weights[0][2]])
dx1 = sum([dvalues[0][0]*weights[1][0], dvalues[0][1]*weights[1][1], dvalues[0][2]*weights[1][2]])
dx2 = sum([dvalues[0][0]*weights[2][0], dvalues[0][1]*weights[2][1], dvalues[0][2]*weights[2][2]])
dx3 = sum([dvalues[0][0]*weights[3][0], dvalues[0][1]*weights[3][1], dvalues[0][2]*weights[3][2]])

dinputs = [dx0, dx1, dx2, dx3]
print(dinputs)


