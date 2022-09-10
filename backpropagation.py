x = [1.0, -2.0, 3.0]
w = [-3.0, -1.0, 2.0]
b = 1.0

# Multiplying inputs by weights
def weighted_sum(x, w, b):
    xw0 = x[0] * w[0]
    xw1 = x[1] * w[1]
    xw2 = x[2] * w[2]
    return xw0 + xw1 + xw2 + b

# Adding weighted inputs and bias
z = weighted_sum(x, w, b)

# ReLU activation 
y = max(z, 0)

# Backward pass

# Derivative from the next layer
dvalue = 1

# Derivative of ReLU and the chain rule
drelu_dz = dvalue * (1. if z>0 else 0.)

# Derivative of sum w.r.t. weighted inputs
dsum_dwx0 = 1
dsum_dwx1 = 1
dsum_dwx2 = 1
dsum_db = 1
# Chain rule i.e. multiplication of derivatives of relu wrt to sum and sum wrt to weighted inputs
drelu_dwx0 = drelu_dz * dsum_dwx0
drelu_dwx1 = drelu_dz * dsum_dwx1
drelu_dwx2 = drelu_dz * dsum_dwx2
drelu_db = drelu_dz * dsum_db

# Partial Derivative wrt to w and x
dmul_dx0 = w[0]
dmul_dx1 = w[1]
dmul_dx2 = w[2]
dmul_dw0 = x[0]
dmul_dw1 = x[1]
dmul_dw2 = x[2]

drelu_dx0 = drelu_dwx0 * dmul_dx0
drelu_dw0 = drelu_dwx0 * dmul_dw0
drelu_dx1 = drelu_dwx1 * dmul_dx1
drelu_dw1 = drelu_dwx1 * dmul_dw1
drelu_dx2 = drelu_dwx2 * dmul_dx2
drelu_dw2 = drelu_dwx2 * dmul_dw2

# The partial derivatives combined into a vectore make up for the gradients.
# The gradients can be represented as:
dx = [drelu_dx0, drelu_dx1, drelu_dw2] # Gradients on inputs
dw = [drelu_dw0, drelu_dw1, drelu_dw2] # Gradients on weights
db = drelu_db   # Gradient on bias (which is 1)

print(w, b)

# For a single neuron layer there is no need of dx
# We only need the gradients on the weights

# Applying fraction of gradients to the original values
w[0] += -0.001 * dw[0]
w[1] += -0.001 * dw[1]
w[2] += -0.001 * dw[2]

print(w, b)

z = weighted_sum(x, w, b)
# ReLU activation
y = max(z, 0)
print(y)





