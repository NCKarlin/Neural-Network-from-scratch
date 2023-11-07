'''
Video 5 in Playlist:
https://www.youtube.com/watch?v=gmjzbpSVY1A&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=5&ab_channel=sentdex

This file contains a little more information to go through or to be explained.

1. INTEGRATION OF A DATA CREATION FUNCTION
-> in order to being able to generate the kind of input data (especially the dimensions)
which we need for the examples, a create_data function has been added
-> it is a spiral dtaa creation method to for uniform class distribution among the points
-> it creates a dataset with specified number of samples per class, each sample being
described by 2 features
-> entire dataset dimensions: points*class (total # of samples) x 2 (features)

2. VISUALIZATION OF THE CREATED DATA
-> if you want to check out the data created, just uncomment the lines for the 
visualisation and you can see the bare points, and the class-color-coded ones

3. SINGLE LAYER PASS WITH ACTIVATION
-> In this example we go back to a simpler example, just passing one single layer with
an additional activation layer behid the neuron outputs
-> Interestingly, one can see, that for the layer creation the dimensions go from 2 to 5,
meaning the input amount of features (2) will be expanded to 5 features per point.
-> Therefore, the output of this layer should have the following dimensions:
[points*classes x 5]
'''
# IMPORTS
import numpy as np #for basic aithmetic operations
import matplotlib.pyplot as plt #for visualization

# Seeding for reproducable randomization
np.random.seed(0)


# Data example creation function (Spiral data point generation)
def create_data(points, classes):
    '''
    This function creates a dataset with the following attributes:
    1. Number of samples per class: points
    2. Number of classes: classes
    3. Number of features per sample: 2 (X- & Y-Coordinate)
    
    Inputs:
    - points: numbers of samples per class for the dataset to be created
    - classes: number of classes for dataset creation
    
    Outputs:
    - X: dataset [points*classes x 2]
    - y: classifications for the created dataset [points*classes x 1]
    '''
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y 


# Dataset creation example (100 samples, 2 features, 3 classes)
X, y = create_data(100, 3) # X = (300, 2) | y = (300,)

# # VISUALIZATION OF CREATED DATASET ####################################################
# # 1. General visualization of data points in created dataset
# plt.scatter(X[:,0], X[:,1])
# plt.show()
# # 2. Class color-coded visualization of data points in created dataset
# plt.scatter(X[:,0], X[:,1], c=y, cmap="brg")
# plt.show()


# Definition of a basic layer class
class Layer_Dense:
    
    # INITIALIZATION (CREATION) of layer and needed values
    def __init__(self, n_inputs, n_neurons):
        # Initializing weight matrix according to batch size and number of neurons
        #! Multiplication with 0.1 to keep initilization values for weights small
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        # Initializing the neuron biases with 0's in the right shape
        self.biases = np.zeros((1, n_neurons))
    
    # FORWARD-PASS through the layer
    def forward(self, inputs):
        # Single layer output creation -> dot-product of inputs and weight + biases
        self.output = np.dot(inputs, self.weights) + self.biases


# Definition of basic ReLU-Activation functionality
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


# Creation of first layer (# of input features -> # of output features/ neurons of layer)
layer1 = Layer_Dense(2, 5)

# Creation of activation layer for current layer output
activation1 = Activation_ReLU()

# Forward-pass through current layer
layer1.forward(X)
# print(layer1.output) # print check for output of current layer (! negative values !)
# print(layer1.output.shape) # print check for layer 1 output shape -> [n_class*pointsx5]

# (Forward-) Pass through activation layer
activation1.forward(layer1.output)
print(activation1.output) #! No zeros should be left
