import numpy as np

softmax_output = np.array([0.7, 0.1, 0.2]
)
softmax_output_reshaped = np.array(softmax_output).reshape(-1, 1)
softmax_output_eyed = softmax_output_reshaped * np.eye(softmax_output_reshaped.shape[0])

# Same results can be achieved by using the diagflat function from numpy

softmax_outputs = np.diagflat(softmax_output_reshaped)

softmax_gradients = softmax_outputs - np.dot(softmax_output_reshaped, softmax_output_reshaped.T)

# print(softmax_output_eyed)
# print(softmax_output)

print(softmax_gradients)
# print(np.diagflat(softmax_output_reshaped) - np.dot(softmax_output_reshaped, softmax_output_reshaped.T))