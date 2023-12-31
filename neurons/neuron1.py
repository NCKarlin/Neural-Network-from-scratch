'''
BASIC NEURON CALCULATION IN RAW PYTHON

Video 1 in playlist: 
https://www.youtube.com/watch?v=Wo5dMEP_BbI&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=2&ab_channel=sentdex

THEORY
The neuron is a if not the central building block of a neural network, therefore we start
with examining it. In this case we imagine a dense/ fully-connected netowrk, which means
that between each layer of neurons iwthin the network, each neuron is connected to all of
the neurons in the previos layer. Depending on the neuron inputs, for each of them the 
neuron holds weights, which are the values by which each neuron input individually is 
multiplied with. Additionally to the values connected to all neuron input, each neuron 
in any neural network also has an additional value which is called bias. After the 
multiplicaiton of the neuron inputs with their respective neuron weights, the bias is
added to the neuron. For now that is all the information needed for this script. The 
reason why both weights and biases are needed in neurla networks will be elaborated 
later in this repository. In more mathematical notation, the output Y of a neuron i can
be expressed as follows:

Yi =  Wi[1] * I1 + Wi[2] * I2 + Wi[3] * I3 + Bi [i denoting the neuron looked at]

THIS SCRIPT
This file contains the code for one executing neuron within the network.
For this specific example, we are imagining a neuron, which has three fully-connected 
neurons in the previous layer feeding into the current neuron.  

Therefore, along the three inputs it receives from the previous neurons, it also
computes with three different weights for each input. Additionally, every unique
neuron in a neural network has it's own unique bias, which is added to the 
calculation of the output.
'''

# Inputs from the previous layer (3 neurons)
inputs = [1, 2, 3]

# Weights for each neuron input (3 neurons -> 3 weights)
weights = [0.2, 0.8, -0.5]

# Unique bias for every unique neuron
bias = 2

# Calculation of output - multiplication of inputs and weights + addition of unique bias
output = inputs[0] * weights[0] + \
         inputs[1] * weights[1] + \
         inputs[2] * weights[2] + \
         bias

print(output)