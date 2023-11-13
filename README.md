# Neural Network from scratch
![](https://github.com/NCKarlin/Neural-Network-from-scratch/blob/main/NN-from-scratch/files/Basic_NN_viz.gif)
[Source: https://towardsdatascience.com/everything-you-need-to-know-about-neural-networks-and-backpropagation-machine-learning-made-easy-e5285bc2be3a]

## Introduction
This repository was instantiated to create a neural network (almost) from scratch and eliminating the use of third-party libraries to dig down deep into understanding the fundamental workings and tuning mechanics of a deep learning neural network. <br>
While it does not walk through the entire cycle of a neural network - the backpropagation is not treated in detail, therefore it stops after the loss determination - it does a good job at explaining the mechanics and mathematical operations that happen within neurons, layers and also when taking batched input into account. <br>
Below you will find the source for this repository, a walk-through of the repository structure and a short disclaimer on the scturcture of the scripts within this repository.

## Disclaimer: Source
All of the code and comments are mainly based on the video tutorial series ["Neural Networks from Scratch"](https://www.youtube.com/watch?v=Wo5dMEP_BbI&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&ab_channel=sentdex), which in turn is based on the book ["Neural Networks from Scratch in Python – Building neural networks in raw Python"](https://nnfs.io/) by Harrison Kinsley and Daniel Kukiela. <br>
I highly recommend this tutorial to anyone who has some basic knowledge about neural networks, but really wants to drill down into the specific mechanics to better understand how to optimise and work with neural networks.

## Run the script in this repo
If one would like to run all the scripts within this repository, since it is mostly independent of third party libraries, not much has to happen. While one could easily run this from the base environment, I still do recommend to create a virtual environment for clarity and good-practice.
```
python -m venv "given-venv-name"
```
After activating the newly created virtual environment the only libraries that have to be installed are the following:
```
pip install numpy #basic arithmetic operations
pip install math #for math constants 
pip install matplotlib #for visualisations
```
Subseqeuntly, fter following these steps everyone should be able to run the scripts within this repository locally.

## Repository Organization
In order to make this as pleasant and accessible as possible, I will shortly explain the ideal chronological order to walk through the repository, while explaining the organization that can be found here. The order of the scripts within the folders do follow numbering, as specific titles would have been too long and confusing. <br>
To follow the order of the tutorial I would suggest to look into the folder and their respective scripts in the following order:
1. **neurons:**<br>
In this folder you will find all scripts that are concerned with the detailed operations of a single neuron (inputs, weights and biases), up until multiple neurons within a layer.
2. **batches:**<br>
In this folder you will find the two scripts concerned with layers of neurons within a network. Firstly, just a batch of inputs for a single layer of multiple neurons and secondly the same detailed operation for a two-layered network with 3 neurons in each layer.
4. **layers:**<br>
In this folder you will find the script that showcases the dynamic layer creation for a neural network with a self-defined layer class, which showcases the most important functions and attributes for a (generic) neural network layer of neurons. 
5. **activation:**<br>
In this folder you will find multiple scripts concerning the activation function, which handels the output of each neuron before passing it on. In this repository the two most common activation functions are walked-through, the **ReLU-activation** for all layers but the output layer, and the **Softmax-activation**, which is mostly used in the output layer of classification networks.
6. **loss:**<br>
In this folder you will find two scripts, which are aimed at showcasing the detailed functionality of the categorical cross-entropy loss, which is the most common loss function for multi-class classification problems. Firstly, the walk-through (loss_categoricalCE.py) and then secondly, the implementation within the simple two-layer network we have designed until then.
7. **visualisations:**<br>
This folder merely contains a simple visualisation for the determination of the derivative of a quadratic function. These mechanics are applied on a vastly different scale for neural networks, but because of the high dimensionality it is impossible to visualize these properly for understanding, hence the 2D example. In real neural networks, this is done with partial derivatives of an intensely multidimensional function, as it is dependent on all the weights and biases within a neural network.

## Script Organization
As for the repo, each script also follows a structure for easability of understanding. Generally, every script should have the following chronological order of components:<br>
1. **TITLE:** <br>
Short descriptive title of the respective script in ALL-CAPS
2. **Video source from tutorial:**<br>
Number of video in playlist and the link to it. 
3. **THEORY:** <br>
This is the section that is used to give theoretical context to the lines of code in the script, mostly it explains the mathematical operations and their implications, so that this theory could be generalized and transferred to other neural networks for other purposes. Disclaimer: some scripts have barely newly introduced theory, therefore the length of this section varies greatly.
4. **THIS SCRIPT:** <br>
This is a short section where the specifics for this script are explained. While the more general theory of it was elaborated on before, this section is more code-oriented and explains in detail what the operations of the specific script are. 
5. **The Code:** <br>
Subsequently, the code for the respective script, which can be run and usually produces printable or displayable results.

## General Comments
Although I had substantial knowledge about neural networks and their functioning, this detailed walk-through helped a lot in solidifying certain aspects but also in identifying new tweaks for better understanding the multitude of operations going on in a neural network. Especially, for tweaking and optimizing neural networks this can be useful, as when the base operations are understood the tweaking options become clear as well. <br>
Furthermore, this repository was created mainly for myself, but I would be delighted if anyone else also finds this helpful. Although (most/ some of) the comments might be unnecessary or too much for clean-coders, I hereby warn people about the amount of commenting, that is part a habit from my side, as well ass a design choice, because I wanted to make this very accessible. <br>
Therefore, I also do not expect anyone to contribute greatly, but if someone would like to, let me know, I'd love to figure out how it could possibly make this simple repository even better. *Thank you!*
