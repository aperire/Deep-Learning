# This code shows the simplified version of neural network where there are two inputs, one hidden node, and one output
import numpy as np 
from numpy import random

# State sigmoid function 
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Hyperparameters
epochs = 500
learning_rate = 0.01

# Put real data when using (random used for test)

# dataset as a list
feature_lst = random.randint(5, size=(10, 2))
label_lst = random.randint(2, size=(10, 1)) #Label should be either 0 or 1


# Weight step, bias initialize to 0
weight_step_hidden_lst = random.randint(1, size=(10, 2)) # size should be same with feature_lst
weight_step_out_lst = random.randint(1, size=(10, 1))

weight_hidden_lst = random.randint(1,2, size=(10, 2))
weight_out_lst = random.randint(1,2, size=(10, 1))

bias_hidden_lst = random.randint(1,2, size=(10, 1))
bias_out_lst = random.randint(1,2, size=(10, 1))



# Calculate list of h values and y hat values
h_lst = np.multiply(feature_lst, weight_hidden_lst) + bias_hidden_lst  #Not summed up yet
h_lst = [[sum(i)] for i in h_lst]
h_lst = np.array(h_lst)

h_sigmoid_lst = []
for i in h_lst:
    for j in i:
        h_sigmoid_lst.append(sigmoid(j))
h_sigmoid_lst = np.array(h_sigmoid_lst)
h_sigmoid_lst = h_sigmoid_lst.reshape(-1, 1)

y_hat_lst = np.multiply(h_sigmoid_lst, weight_out_lst) + bias_out_lst
y_hat_lst = [[sum(i) for i in y_hat_lst]]
y_hat_lst = np.array(y_hat_lst)

y_hat_sigmoid_lst = []
for i in y_hat_lst:
    for j in i:
        y_hat_sigmoid_lst.append(sigmoid(j))
y_hat_sigmoid_lst = np.array(y_hat_sigmoid_lst)
y_hat_sigmoid_lst = y_hat_sigmoid_lst.reshape(-1, 1)

# Calculate Error Gradient (output unit)
delta_output = (label_lst - y_hat_sigmoid_lst) * y_hat_sigmoid_lst * (1 - y_hat_sigmoid_lst)

# Propagate Errors to hidden layer
delta_hidden = delta_output * weight_out_lst * h_sigmoid_lst * (1 - h_sigmoid_lst)

# Update Weight Steps

weight_step_out_lst = weight_step_out_lst.astype(np.float32) #Convert to float to add with floats
weight_step_out_lst += delta_output * h_sigmoid_lst

# Duplicate delta values by the number of inputs for distribution
delta_hidden = delta_hidden.tolist()
for i in delta_hidden:
    if len(i) != 2:
        i.append(i[0])
delta_hidden = np.array(delta_hidden)

weight_step_hidden_lst = weight_step_hidden_lst.astype(np.float32)
weight_step_hidden_lst += delta_hidden * feature_lst
print(f"delta output : {delta_output}\ndelta hidden : {delta_hidden}")

# Update weights
weight_out_lst = weight_out_lst.astype(np.float32)
weight_hidden_lst = weight_hidden_lst.astype(np.float32)

weight_out_lst += weight_step_out_lst * learning_rate
weight_hidden_lst += weight_step_hidden_lst * learning_rate














