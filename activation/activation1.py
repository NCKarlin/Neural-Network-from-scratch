import numpy as np

'''
Video 5 in Playlist:
https://www.youtube.com/watch?v=gmjzbpSVY1A&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=5&ab_channel=sentdex


This file contains the small piece of code, which showcases the behaviour of activation
functions. 

In this case we only look at the activation function step within the network. This means
that the outputs from the previous layer neurons have already been calculated (inputs),
and the activation function in form of ReLU (x<=0 -> 0 | x > 0 -> x) is modeled on those.

Therefore, we expect the output to be a list of the same length as inputs, but with all
the values below 0 being truncated to 0. 
'''

# Neuron outputs from previous layer as input for this layer
inputs = [0.0, 2.0, 3.3, -2.7, 1.1, 2.2, -100]

# Empty output list to append results to
output = []

# ReLU as for-loop (example 1)
for i in inputs:
    if i > 0:
        output.append(i)
    elif i <= 0:
        output.append(0)

# ReLU as for-loop (example 2)
for i in inputs:
    output.append(max(0, i))
    

print(output)