'''
Video 4 in Playlist:
https://www.youtube.com/watch?v=TEWy9vZcxW4&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=6&ab_channel=sentdex

This file contains the small piece of code, which essentially showcases the same 
operation as in batches2.py, but this time iwth the difference of creating a class 
for the layer objects, so they can be instantiated dynamically and passed through
with simple and easy commands. 

#! Take a look at the sizes of the resulting layer outputs
-> Interestingly, one can see that the layer output is governed by the number of samples
in each batch that is being passed through, as well as the number of neurons in the 
respective layer.
-> Therefore, layer 1's output is of the shape: 3 (# inputs) x 5 (# neurons)
-> Therefore, layer 2's output is of the shape: 3 (# inputs) x 2 (# neurons)

#! Output values will change due to random weights and biases initialization

Below, the mechanics of the dot product performed here:

INPUTS (3,4)        NEURONS (4,3)   OUTPUTS (3,3)

a11 a12 a13 a14     b11 b12 b13     [(a11*b11)+(a12*b21)+(a13*b31)+(a14*b41)]  [(a11*b12)+(a12*b22)+(a13*b32)+(a14*b41)]  [(a11*b13)+(a12*b23)+(a13*b33)+(a14*b43)]
a21 a22 a23 a24  x  b21 b22 b23  =  [(a21*b11)+(a22*b21)+(a23*b31)+(a24*b41)]  [(a21*b12)+(a22*b22)+(a23*b32)+(a24*b42)]  [(a21*b13)+(a22*b23)+(a23*b33)+(a24*b43)]
a31 a32 a33 a34     b31 b32 b33     [(a31*b11)+(a32*b21)+(a33*b31)+(a34*b41)]  [(a31*b12)+(a32*b22)+(a33*b32)+(a34*b42)]  [(a31*b13)+(a32*b23)+(a33*b33)+(a34*b43)]
                    b41 b42 b43
'''
# IMPORTS
import numpy as np #for basic arithmetic operations

# Seeding for reproducable randomization
np.random.seed(0)

# Inputs from previous layer of the batch (N=3)
X = [[1.0, 2.0, 3.0, 2.5],
     [2.0, 5.0, -1.0, 2.0], 
     [-1.5, 2.7, 3.3, -0.8]]

# Definition of a basic layer class
class Layer_Dense:
    
    # INITIALIZATION (CREATION) of layer and needed values
    def __init__(self, n_inputs, n_neurons):
        # Initializing weight matrix according to batch size and number of neurons
        #! Mulitplication with 0.1 to keep initilization values for weights small
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        # Initializing the neuron biases with 0's in the right shape
        self.biases = np.zeros((1, n_neurons))
    
    # FORWARD-PASS through the layer
    def forward(self, inputs):
        # Single layer output creation -> dot-product of inputs and weight + biases
        self.output = np.dot(inputs, self.weights) + self.biases

# Layer creation (Initialization of Layer_Dense object)
layer1 = Layer_Dense(4, 5)
layer2 = Layer_Dense(5, 2)

# Forward-pass through first layer
layer1.forward(X)
print(layer1.output) # n_inputs [3] x n_neurons [5]
print(layer1.output.shape)

# Forward-pass through second layer
layer2.forward(layer1.output)
print(layer2.output) # n_inputs [3] x n_neurons [2]
print(layer2.output.shape)