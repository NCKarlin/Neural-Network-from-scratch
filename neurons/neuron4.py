'''
Video 3 in Playlist: 
https://www.youtube.com/watch?v=tMrbN67U9d4&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=3&ab_channel=sentdex

This file contains the small piece of code, which essentially does the same operation as 
in neuron1.py, where we are just looking at the output of a single neuron within a layer.

But again, instead of manually creating the output calculation, this time the built-in 
dot-product function of numpy is used. 

The dot-product function performs exactly the operation we want to undertake for the 
weights and the inputs, as it element-wise will multiply and then add the result of
those multiplications together to create a scalar (single) value.
'''
# IMPORTS
import numpy as np #for basic arithmetic operations

# Inputs from the neurons of the previous layer
inputs = [1, 2, 3, 2.5]

# Weights for the individual neuron in the current layer
weights = [0.2, 0.8, -0.5, 1.0]

# Bias for the individual neuron in the current layer
bias = 2

# Determining the output through matrix/ dot product
output = np.dot(weights, inputs) + bias

print(output)