'''
NEURON LAYER OUTPUT CALCULATION WITH NP.DOT(...)

Video 3 in Playlist: 
https://www.youtube.com/watch?v=tMrbN67U9d4&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=3&ab_channel=sentdex

THEORY
There is no great network theory involved in here once more, but rather some programming
theory, as it showcases the output calculation of a layer of neurons as in neuron3.py, 
with the main difference of the direct implementation of np.dot(...), which as you can
see saves quite some space in comparison with the nested for-loop.

THIS SCRIPT
This file contains the small piece of code, which essentially does the same as the script
in neuron3.py, hence the resulting values should also be the same. 
The main difference for this file, specifically in contrast to neuron4.py, is that 
instead of performing the input-weight multiplication and the bias addition within
a for-loop, now "only" the dot product function from numpy is used. 

#!
Here it is important to consider the order/ sequence of the inputs, as the first 
element of the input will determine the way of indexing and performing the dot-product.
By leading with the weights (as these are individual for every neuron), while the inputs
in a fully connected network are the same for all neurons, the weights and the inputs
are multiplied accordingly for every neuron, by using the dot-product on the inputs 
instead of looping through. 
#!
'''
# IMPORTS
import numpy as np #for basic arithmetic operations

# Inputs from the neurons of the previous layer
inputs = [1, 2, 3, 2.5]

# List of list of weights for neuron input of current layer
weights = [[0.2, 0.8, -0.5, 1.0], 
           [0.5, -0.91, 0.26, -0.5], 
           [-0.26, -0.27, 0.17, 0.87]]

# List of biases for each neuron of current layer
biases = [2, 3, 0.5]

# Calculating the output of the current layer
#! weights have to be passed first for indexing the dot product (we want 3 values -> 3 neurons)
output = np.dot(weights, inputs) + biases

print(output)