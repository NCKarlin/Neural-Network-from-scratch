'''
CATEGORICAL LOSS CALCULATION IN (ALMOST) RAW PYTHON

Video 7 in Playlist:
https://www.youtube.com/watch?v=dEXPMQXoiLc&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=7&ab_channel=sentdex

THEORY
After a pass through a NN and arriving at output values for each sample, which are 
probabilities representing the class prediction, we now need some measure to evaluate
how fitting or unfitting these predicted class probabilities are. 
For this purpose a loss function is defined, and depending on the problem, different 
loss-functions will make more or less sense, even some requiring hand-crafter loss 
functions. In this case, for a classification problem, one of the best performing loss-
functions is the categorical cross-entropy loss, which is defined like this:

#! Value ranges
prediction: [0, 1]
loss: (-inf, 0]

L[i] = - Sum[j](target[i,j]*log(prediction[i,j]))   
i: index for sample
j:index for class-prediction

#! One-hot encoding for target vector
The target vector is a one-hot-encoded vector for the correct label. This means that at 
the index-location of sample prediction, the prediction vector will be 1, otherwise 0.
This has the great advantage, that all multiplication terms of prediction with target, 
which are not the target class will fall out, as theyr're multiplied by 0, while the 
values of logaritm of the other values remains untouched as the target equals 1.
Therefore, the actual loss calculation can also be done just by taking the negative
logaritm of the respective class probability of the correct class for that sample. 

Interestingly, when thinking about the mechanics of these operations, the higher the 
confidence in the prediction probability by the model ([0, 1]), the higher the value fed
into the logarithm. Since the logaritm of values between 0 and 1 always is negative, 
we multiply the result by -1 and convert to the positive range. 
The closer the values of the predictions are to 1, the closer the logaritm of them will
be to 0. In contrast, the closer the prediction is to 0, the greater the positive 
resulting value will be as it is multiplied with -1.
-> MEANING: The more confidence in the prediction (given it is correct) the lower the 
loss in comparison to the loss resulting from wrong predictions, with lower probability
values.

THIS SCRIPT
In this script we will jsut simply perform the operations of the categorical cross-
entropy loss function, and additionally show, that the expansive calculation comprising
all softmax outputs, yields the same result as the "simple" negative logaritm of the 
prediction probability of the correct class.
'''
# IMPORTS
import math #for log-function

# Exempalry softmax activation output for one sample and three classes
softmax_output = [0.7, 0.1, 0.2]

# Exemplary one-hot-encoded target values
target_output = [1, 0, 0]

# Catgerocial Cross-Entropy loss calculation
loss = -(math.log(softmax_output[0])*target_output[0]+ # 0.7 * 1
         math.log(softmax_output[1])*target_output[1]+ # 0.1 * 0 -> falls out
         math.log(softmax_output[2])*target_output[2]) # 0.2 * 0 -> falls out

# Essential calculation going on above
loss_calc = -math.log(softmax_output[0])

print(loss)
print(loss_calc)