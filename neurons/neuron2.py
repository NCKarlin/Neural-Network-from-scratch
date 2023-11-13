'''
NEURON OUTPUT CALCULATION FOR ENTIRE LAYER IN RAW PYTHON

Video 2 in Playlist: 
https://www.youtube.com/watch?v=lGLto9Xd7bU&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=2&ab_channel=sentdex

THEORY
As neural networks never consist of only one neuron but rather of several layers of 
neurons, as a next step the calculations we performed for a single neuron in the script
neuron1.py are noe being modeled for a layer of fully-connected neurons. 
Therefore, we can imagine each neuron within the current layer receiving information 
from all the neurons of the previous layer. This means, that the input to each neuron
in the current layer is the same, while the individual weights for the inputs at each
neuron differ, just as the individual bias. 
There is not too much new theory for the general working of neural networks, because 
this is rather focused on how the multiplication can be carried out correctly.

THIS SCRIPT
In this example the layer we are looking at a neuron layer consistsing of 3 neurons and
4 inputs each, from the four neurons in the previous layer. 
As in neuron1.py the mechanics of the calculation stay the same, with the main 
difference this time the output will not only be a single value but a list of 
three values as each neuron computes an output itself within the layer.
'''

# Inputs from the neurons of the previous layer
inputs = [1, 2, 3, 2.5]

# Weights for each neuron and the respective inputs
weights1 = [0.2, 0.8, -0.5, 1.0]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]

# Unique biases for the distinct neurons of the layer
bias1 = 2
bias2 = 3
bias3 = 0.5

# Calculation of the output of all three neurons
output = [inputs[0]*weights1[0] + inputs[1]*weights1[1] + inputs[2]*weights1[2] + inputs[3]*weights1[3] + bias1, #neuron 1
          inputs[0]*weights2[0] + inputs[1]*weights2[1] + inputs[2]*weights2[2] + inputs[3]*weights2[3] + bias2, #neuron 2
          inputs[0]*weights3[0] + inputs[1]*weights3[1] + inputs[2]*weights3[2] + inputs[3]*weights3[3] + bias3,] #neuron 3

print(output)