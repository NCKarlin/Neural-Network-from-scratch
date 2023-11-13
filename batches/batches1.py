'''
LAYER OUTPUT CALCULATION FOR BATCHES

Video 4 in Playlist:
https://www.youtube.com/watch?v=TEWy9vZcxW4&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=6&ab_channel=sentdex

THEORY
Neural networks most often are trained on multi-core machine machines, which means they 
have multiple processors, that can carry out some calculations in parallel. While you
could just simply hand over one sample at a time and let the network step through all 
samples, it would be painfully slow, and while you most of the time not can hand over 
the entire dataset, because that would require too much memory, usually the dataset
is divided into batches of samples, that can be computed in parallel and do not overload
the memory. Hence, here the layer output calculation here is being carried out for a
batch of values. The central part of theory for this calculation is the mathematical 
form, which is explained below.

THIS SCRIPT
This file contains the small piece of code, which shows how the dot product of the 
neuron weights and the respective neuron inputs from the previous layer is carried out
in the case of several samples being handed over within a batch. 
Therefore, we now add some "rows"/ lists to the inputs list, representing the inputs from
the neurons of the previous layer for a batch size of three samples. Leaving the output
calculation as it was would lead to a shape error for the output calculation, because now
both the inputs as well as the weights have the exact same shape: [(3, 4)].
But since we want to multiply and add the results of the inputs of the batch with the 
respective weights, at least one of the inputs to the dot-product have to be transformed. 
This means, that by transposing the weights (from row=neuron | column=weights for inputs
of neuron -> row: weights for inputs of neuron | column=neuron), we will make sure, that
the correct inputs are multiplied with the correct weights to generate the correct 
output.

#! Only numpy arrays can be transposed, NO lists -> conversion to np.array(...)
'''
# IMPORTS
import numpy as np #for basic airthmetic operations

# Inputs from previous layer of the batch (N=3)
inputs = [[1.0, 2.0, 3.0, 2.5],
          [2.0, 5.0, -1.0, 2.0], 
          [-1.5, 2.7, 3.3, -0.8]]

# List of individual neuron weights of current layer
weights = [[0.2, 0.8, -0.5, 1.0], 
           [0.5, -0.91, 0.26, -0.5], 
           [-0.26, -0.27, 0.17, 0.87]]

# List of individual neuron biases of current layer
biases = [2, 3, 0.5]

# Calculation of the current layer output for the batch
#! Reversed order and weight tranposition for correct dot product calculation
output = np.dot(inputs, np.array(weights).T) + biases

print(output)
