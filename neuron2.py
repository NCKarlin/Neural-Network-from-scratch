'''
Video 2 in Playlist: 
https://www.youtube.com/watch?v=lGLto9Xd7bU&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=2&ab_channel=sentdex

Now as a next step we will not only model one single (random) neuron from the network, 
but we will model one output geeneration for a simply structured layer of neurons. 

In this example the layer we are looking at consists of 3 neurons and 4 inputs each, 
from the four neurons in the previous layer. 

As in neuron1.py the mechanics of the calculation stays the same, with the main 
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