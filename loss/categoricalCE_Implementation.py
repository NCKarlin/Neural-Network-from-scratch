'''
CATEGORICAL LOSS IMPLEMENTATION FOR SIMPLE DENSE NEURAL NETWORK

Video 8 in Playlist:
https://www.youtube.com/watch?v=levekYbxauw&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=8&ab_channel=sentdex

THEORY
There is not too much added theory in this script as it is mostly about how to integrate
the previously mentioned categorical cross-entropy loss into the neural network frame-
work. One theoretical addition is the clipping of the softmaxed network output 
(class prediction probabilities), before calculating the loss.

#! Why prediction probability clippping (softmaxed network output)?
The main reaason for clipping the softmaxed network outputs in the form of prediction
probabilities is, that one desperately wants to avoid prediction probability values of
0. The natural log of 0 of infinity, which would lead to the loss value of such a 
prediction being -inf. Subsequently, when trying to determine the batch mean loss, 
the mean loss value would be -inf as average.
Therefore, one performs clipping of the softmaxed network output values, which means 
that one clips all the output values at a very smalle value (1e-7) and in order to not
incorporate any bias one also clips them at the upper end (1-1e-7). Thus, no loss value
can become -inf.

THIS SCRIPT
While the former script was more about the calculation of the loss, this script rather 
focuses on the integration of the loss calculation into a neural network model. That's 
why, two new classes are defined:
1. General Loss Class:
-> The general (parent) Loss class has only one method, which is to calculate the loss, 
according to the more refined loss definition.
2. Categorical Cross Entropy Loss:
-> The more specific Cross Entropy Loss class has, besides the inherited calculate 
function and additional function named forward. In that function the following happens:
    2.1 Clipping of the softmaxed network outputs between 0 and 1
    2.2 Determination of target/ ground truth encoding
        -> list of single indices of true classes
        -> one-hot-encoded list/ array of true class membership
    2.3 Negative log of true target class prediction values for loss calculation
    2.4 Returning batch categorical cross entropy loss values 
All of this is integrated after the network setup, as in the script 
activation6_Softmax.py. 

'''

# IMPORTS
import numpy as np #for basic airhmetic operations

# Basic Spiral Data Creation - commented version can be found in: activation2_ReLU.py
def create_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y 

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


# Definition of basic Softmax-function as class - commented: activation6_Softmax.py
class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


# General loss definition as class
class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        batch_losses = np.mean(sample_losses)
        return batch_losses


# Categorical Cross Entropy Loss Definition as class 
class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        # Number of samples within the batch
        samples = len(y_pred)
        # Clipping values - for no 0 value predictions and upper limit for symmetry
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        # Ground truth data single dimension list with correct class index
        if len(y_true.shape) == 1:
            # Slicing the clipped values to retireve correct confidence
            correct_confidences = y_pred_clipped[range(samples), y_true]
        # Ground truth data as one hot encoded multidimensional array
        elif len(y_true) == 2:
            # Summing over the multiplication with multidimensional ground truth array
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
        # Calculating negative log values as losses for correct confidences
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

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

# LOSS CALCULATION ######################################################################
# Instantiating loss and calculating btach losses
loss_function = Loss_CategoricalCrossEntropy()
loss = loss_function.calculate(activation2.output, y)