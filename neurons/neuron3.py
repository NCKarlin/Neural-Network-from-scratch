'''
NEURON LAYER OUTPUT CALCULATION IN RAW PYTHON (FOR-LOOPS)

Video 3 in Playlist: 
https://www.youtube.com/watch?v=tMrbN67U9d4&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=3&ab_channel=sentdex

THEORY
There is no new theory involved here, except programming theory on how to structure
the neuron layer output calulation more efficiently.

THIS SCRIPT
This file contains essentially the same operation as performed in neuron2.py, with the 
difference, that instead of manually calculating the output, the dataformats are adjusted,
so that the calculation can be performed within a nested for-loop.
The first for-loop will loop over the neurons in our current layer receiving the inputs
from the previous layer, while the second nested for-loop will then actually compute the
individual product of the respective neurons weights with the inputs.
'''

# Inputs from the neurons of the previous layer
inputs = [1, 2, 3, 2.5]

# List of list of weights for neuron input of current layer
weights = [[0.2, 0.8, -0.5, 1.0], 
           [0.5, -0.91, 0.26, -0.5], 
           [-0.26, -0.27, 0.17, 0.87]]

# List of biases for each neuron of current layer
biases = [2, 3, 0.5]

# Calculation of layer output
layer_outputs = [] #placeholder for current layer output
# Iterating over all neurons
for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = 0 #output of each individual neuron
    # Iterating over individual neuron inputs and weights
    for n_input, weight in zip(inputs, neuron_weights):
        # Multiplication of input with respective weight
        neuron_output += n_input * weight
    # Addition of indivual neuron bias to weighted inputs
    neuron_output += neuron_bias
    # Appending if to placeholder list for neuron outputs
    layer_outputs.append(neuron_output)

print(layer_outputs)