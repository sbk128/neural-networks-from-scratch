x = [1.0, -2.0, 3.0]
w = [-3.0, -1.0, 2.0]
b = 1.0

# Multiplying inputs by weights
xw0 = x[0] * w[0]
xw1 = x[1] * w[1]
xw2 = x[2] * w[2]

# Adding weighted inputs and bias
z = xw0 + xw1 + xw2 + b

# ReLU activation 
y = max(z, 0)

# Backward pass

# Derivative from the nexr layer
dvalue = 1

# Derivative of ReLU and the chain rule
drelu_dz = dvalue * (1. if z>0 else 0.)
print(drelu_dz)
