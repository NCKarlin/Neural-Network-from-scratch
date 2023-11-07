'''
SOTMAX ACTIVATION WITH NUMPY FUNCTIONS

Video 6 in Playlist:
https://www.youtube.com/watch?v=omz_NdFgWyU&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=6&ab_channel=sentdex

THEORY
There is no additional theory in this script, compared to activation3-Softmax.py.
The main difference is the implementation of numpy functions for basic arithmetic 
operations, which saves space, lines and also makes for better readbility.

THIS SCRIPT
This script contains the small piece of code with the same functionality as activation3-
Softmax.py, executed with numpy functions instead of raw python.
'''
# IMPORTS
import math #for Euler number (e)
import numpy as np #for basic airhmetic operations

# Example layer outputs from the previous layer
layer_outputs = [4.8, 1.21, 2.385]

# Euler number -> 2.7182818...
e = math.e

# Exponentiate layer outputs
exp_values = np.exp(layer_outputs)

# Normalize the exponentiated layer outputs
norm_values = exp_values / np.sum(exp_values)

print(norm_values)
print(np.sum(norm_values)) #should be 1/ 0.99999...