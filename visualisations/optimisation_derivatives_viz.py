'''
VISUALISATION OF DERIVATIVE TANGENT LINES FOR MULTIPLE POINTS OF A FUNCTION

Video 9 in Playlist:
https://www.youtube.com/watch?v=txh3TQDwP1g&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=9&ab_channel=sentdex

THEORY
This is mainly just a visualisation script, which is supposed to help understand the 
basic workings of derivatives on functions. 
Generally, in a neural networks the tweaking of the weights and biases on the neurons
is an action of which one would like to know the impact onto the ultimate loss of the
respective model value constellation. Therefore, the derivative is ideal to give 
information on how the function is going to change at that point depending on where and 
how you move the values for the weights and biases.
Hence, by asbtracting a bit more and the utilization of the chain rule, this does not
have to be done for every weight and every bias for every sample to determine on how to
adjust the model best, but instead one works with partial derivatives, which are more 
efficient at determining in which direction each neuron's weights and biases have to 
change to mimise the loss function fo the respective model.

THIS SCRIPT
This script is just a relatively simple visualization of derivative tangent lines for a
quadratic function. Currently, it does this visualization for the following function:
f(x) = 2 • x • x
within the value range x = [0, 6]
only visualizing the tangent-line for a range of x-values +-0.9 around the point
'''

import matplotlib.pyplot as plt
import numpy as np 

# f(x) = 2•x•x
def f(x):
    return 2*x**2

# Definition of a range of values for f(x)
x = np.arange(0,6,0.001)
y = f(x)

# Plotting of the regular function
plt.plot(x, y, c='orange')

# Definition of color array for points and tangents to be plotted
colors = ['k', 'g', 'r', 'b', 'c']

# Function definition for determining function of tangent line
def approximate_tangent_line(x, approximate_derivative, b):
    return approximate_derivative*x + b

# For-loop walking through five points with tangent line exmaples to be plotted
for i in range(5):
    # Determining infinitesimally close second point for approximate numerical derivative 
    p2_delta = 0.0001
    x1 = i
    x2 = x1 + p2_delta #2.0001
    y1 = f(x1) #8
    y2 = f(x2) #8.00080002

    print((x1, y1), (x2, y2))

    # Approximation of numerical derivative for (x1, y1) and (x2, y2)
    approximate_derivative = (y2 - y1) / (x2 - x1)
    # Determining y-axis interceipt from y = m•x + b -> b = y - m•x
    b = y2 - approximate_derivative*x2

    # Definition of x-value range for plotting the tangent line
    to_plot = [x1-0.9, x1, x1+0.9]
    
    # Scattering the points for the tangent
    plt.scatter(x1, y1, c=colors[i])

    # Plotting fo the tangent line: x=to_plot (x-value range) | y = m•xi + b for every xi
    plt.plot(to_plot, 
             [approximate_tangent_line(point, approximate_derivative, b)\
                 for point in to_plot], 
             c=colors[i])

    # Print statement 
    print('Approximate derivative for f(x)',
        f'where x = {x1} is {approximate_derivative}')

plt.show()