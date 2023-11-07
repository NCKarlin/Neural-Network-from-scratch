'''
Video 4 in Playlist:
https://www.youtube.com/watch?v=TEWy9vZcxW4&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=6&ab_channel=sentdex

This file contains the small piece of code, which showcases the output calculation for 
an input of a batch size with three samples, and two layers of neurons, processing the 
input.

In contrast to the examples before, where we've just looked at one single layer of 
neurons before, now it is going to be two layers of each three neurons processing. 
Firstly, the given input (3 samples with 4 features) are fed to the first layer to create
the intermediate output of layer 1 (3 samples with 3 neuron/layer outputs).
Secondly, this intermediate output is fed into the second layer, which consists of 3
neurons with 3 weights for the respective layer 1 outputs.

The dot-product of the intermediate results with the corresponding weights and biases
of layer 2 then create the final output of the second layer. 
'''
# IMPORTS
import numpy as np #for basic arithmetic operations

# Inputs from previous layer of the batch (N=3)
X = [[1.0, 2.0, 3.0, 2.5],
     [2.0, 5.0, -1.0, 2.0], 
     [-1.5, 2.7, 3.3, -0.8]]

# LAYER 1 ###############################################################################
# List of individual neuron weights of first layer
weights1 = [[0.2, 0.8, -0.5, 1.0], 
            [0.5, -0.91, 0.26, -0.5], 
            [-0.26, -0.27, 0.17, 0.87]]
# List of individual neuron biases of first layer
biases1 = [2, 3, 0.5]

# LAYER 2 ###############################################################################
# List of individual neuron weights for second layer
weights2 = [[0.1, -0.14, 0.5], 
            [-0.5, 0.12, -0.33], 
            [-0.44, 0.73, -0.13]]
# List of individual neuron biases of second layer
biases2 = [-1, 2, -0.5]

# Calculation of the outputs from layer 1
layer1_outputs = np.dot(X, np.array(weights1).T) + biases1
print(layer1_outputs.shape)

# Calculation of the outputs from layer 2
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

print(layer2_outputs)