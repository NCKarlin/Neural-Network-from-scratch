'''
TWO-LAYER (INPUT-OUTPUT) PASS OF BATCHED SAMPLES 

Video 6 in Playlist:
https://www.youtube.com/watch?v=omz_NdFgWyU&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=6&ab_channel=sentdex

THEORY
This script covers an exemplary pass of a very simple network of the following structure:
0. input (2 features) -> 2 coordinates
1. input layer (expansion from 2 to 3 features) -> 3 hidden features
2. ReLU-activation of input layer output
3. Output layer (contraction from 3 to number of classes (3)) -> 3 class-output features
4. Softmax-activation of the output-layer output -> 3 classification probabilities

#! OVERFLOW-PREVENTION
Since the softmax function uses exponentiation in its functionality, there is one 
difficulty to handle: in the case of positive layer output values exponentiation very
quickly leads to very large numbers, which can overflow the machines memory. 
Therefore, in the class definition of Activation_Softmax the layer output values, which
are fed into the activation function are shifted by the overall maximum value. This means
that the biggest value resulting after the shift will be 0, while all other values are 
negative. Exponentiating any negative value will give you a result within the range of
0 to 1. Hence, by shifting all values to the negative range before exponentiating, 
prevents the values from becoming too large for the machine memory for continued training.
This step does not have any impact on the results of the normalization afterwards.

THIS SCRIPT
In this script contains the creation of the simple network described above, with each 
layer defined as a class. 
'''
# IMPORTS
import numpy as np #for basic airhmetic operations
from activation2_ReLU import create_data #for spiral dtaa creation

# Basic Dense Layer Class - commented version can be found in activation2-ReLU.py
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


# Definition of basic ReLU-Activation functionality as class
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


# Definition of basic Softmax-function as class
class Activation_Softmax:
    def forward(self, inputs):
        # Exponentiation and maximum shifting to protect from overflow-values
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalization
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        # Defining the probabilities as class attribute of softmax activation class
        self.output = probabilities


# Creation of input data
X, y = create_data(points=100, classes=3)

# DEFINITION OF NETWORK LAYERS ##########################################################
# Definition of the first dense layer (input-layer)
dense1 = Layer_Dense(2, 3) #2 input features -> 3 hidden features (through 3 neurons)
# Definition of the first ReLU-activation layer after first dense network layer
activation1 = Activation_ReLU()
# Definition of the second dense layer as network output layer
dense2 = Layer_Dense(3, 3) #3 from previous layer to 3 classes
# Definition of the output activation layer as softmax activation
activation2 = Activation_Softmax()

# EXAMPLARY NETWORK PASS WITH DATA ######################################################
# Layer 1 | Input layer
dense1.forward(X)
activation1.forward(dense1.output)
# Layer 2 | Output layer
dense2.forward(activation1.output)
activation2.forward(dense2.output)

# Showing the first 5 samples of resulting output
print(activation2.output[:5]) #almost perfectly 1/3 for all -> random initialisation