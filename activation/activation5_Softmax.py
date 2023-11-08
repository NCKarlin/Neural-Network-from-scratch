'''
SOFTMAX ACTIVATION WITH BATCHED LAYER OUTPUT

Video 6 in Playlist:
https://www.youtube.com/watch?v=omz_NdFgWyU&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=6&ab_channel=sentdex

THEORY
There was no new theory added, besides some functionality theory of numpy operations.

THIS SCRIPT
This script contains the small piece of code, which again displays the same functionality
as activation3/4-Softmax.py, this time demonstrating the workings for layer outputs in
batches.  
'''
#IMPORTS
import numpy as np #for basic airhmetic operations

# Example layer outputs from the previous layer for a batch (N=3)
layer_outputs = [[4.8, 1.21, 2.385], 
                 [8.9, -1.81, 0.2], 
                 [1.41, 1.051, 0.026]]

# Exponentiate layer outputs
exp_values = np.exp(layer_outputs)

# Normalize the exponentiated layer outputs
norm_values = exp_values / np.sum(exp_values, 
                                  axis=1, #for summing along rows / along sample
                                  keepdims=True) #for keeping orientation for division

print(norm_values)
print(np.sum(norm_values, axis=1, keepdims=True)) #should be 1/ 0.99999...