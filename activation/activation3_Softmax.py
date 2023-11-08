'''
SOFTMAX ACTIVATION IN RAW PYTHON

Video 6 in Playlist:
https://www.youtube.com/watch?v=omz_NdFgWyU&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=6&ab_channel=sentdex

THEORY
In the last layer of neurons of the network, the subsequent activation layer/ function 
differs from the mostly used ReLU-activation function. The reason a different activation
function is used, is because the network outputs from the last layer should give some 
form of meaning for the next steps. This means, their negative values range should 
give meaning, while the outputs should be converted to be of some form of probability for 
classification tasks for example. Therefore the raw network output is exponentiated, 
to keep all network outputs positive, while negative values are between 0 and 1 and 
positive values are multiples.
These values, are then normalized with by the sum of all layer outputs, so that the 
resulting values are uniformly inbetween 0 and 1 and in the case of classification tasks
they do represent the probability to be part of the corresponding classes, hence they
sum up to 1. 
By doing this, not only do the values of one sample give meaning in relation to another
(the higher the value the higher the chance for the sample to be part of a specific 
class), but in also in relation to the other sample predictions (how clearly does the
model/ network distinguish between two different classes for different samples).
Therefore, as the activation function for the last neuron layer serves a different 
purpose it often differs from the activation function utilized in the model, and 
currently - at least for classificiation tasks - the softmax-function is the most 
popular choice. 

THIS SCRIPT
This file contains the small piece of code showcasing the functionality of the softmax-
activation of layer outputs in raw python (could be entirely raw by hard-coding e).
As the functionality of the softmax-function is to exponentiate the layer outputs and
subsequently normalizing these values, these are also the steps undertaken with the 
example values below.
'''
# IMMPORTS
import math #for Euler number

# Example layer outputs from the previous layer
layer_outputs = [4.8, 1.21, 2.385]

# Euler number -> 2.7182818...
e = math.e

# placeholder for exponentiated values
exp_values = [] 

# For-loop to exponentiate layer output values
for output in layer_outputs:
    exp_values.append(e**output)

# Normalization of the exponentiated layer output values
norm_base = sum(exp_values) #norma base for division
norm_values = [] #placeholder for results
for value in exp_values:
    norm_values.append(value / norm_base)

print(norm_values)
print(sum(norm_values)) # should be 1/ 0.999999999...

