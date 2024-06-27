# Chapter 5 : Calculating Network Error with Loss

Pg. 111-130

## Subsections
1. Introduction
2. Categorical Cross-Entropy Loss
3. The Categorical Cross-Entropy Loss Class
4. Combining everything up to this point
5. Accuracy Calculation

# Introduction

The Loss function, also known as Cost Function, quantifies how wrong the model is. Counterintutively, we want the loss to be as loss as possible (0).

# Categorical Cross-Entropy Loss

The Categorical Cross-Entropy Loss is a loss function that is used in multi-class classification problems. Always used for models that output layer has Softmax Activation. Most common loss function for models that output a probability distribution over a set of classes.

```python
model_output = [0.1, 0.1, 0.6, 0.1, 0.1]

```
Categorical Cross Entropy compares two probability distrubutions, the predicted probability distribution and the actual probability distribution. The actual probability distribution is a one-hot encoded vector.

```
Equation: -sum(y_i * log(y_hat_i))

Simplified to : Li = -log(correct_class_probability)
```

```python
import math

# example output from output layer of nn
softmax_output = [0.7, 0.1, 0.2]
# ground truth
target_output = [1, 0, 0]

print(f"Loss: {-(math.log(softmax_output[0])*target_output[0])}")
```
```
Loss: 0.35667494393873245
```

## Problem #1 - Handling Batches

When we have a batch of data, we have to calculate the loss for each sample in the batch and then average the loss.

```python
# average loss per batch - numpy

neg_log = -np.log(softmax_outputs[
    range(len(softmax_outputs)), class_targets
])
print(f"Losses : {neg_log}")

# average loss
average_loss = np.mean(neg_log)
print(f"Average Loss: {average_loss}")
```

## Problem #2 - Avoiding Division by Zero
We need to avoid division by zero when calculating the loss. We can do this by clipping the values of the softmax output.

```python
# use numpy to clip
y_pred = [0, 1, 0]

y_pred_clippped = np.clip(y_pred, 1e-7, 1-1e-7)
print(y_pred_clippped)
```

```
[1.0000000e-07 9.9999999e-01 1.0000000e-07]
```



# The Categorical Cross-Entropy Loss Class

```python
# common loss class
class Loss:

    # calculates data and regularization losses
    # given model output and ground truth values
    def calculate(self, output, y):

        # calculate sample losses
        sample_losses = self.forward(output, y)
        print("Sample Losses: ", sample_losses)

        # calculate mean loss
        data_loss = np.mean(sample_losses)
        print(f"Data Loss: {data_loss}")

        # return loss
        return data_loss
```

```python
# cross entropy loss
# inherits from loss class ****
class Loss_CategoricalCrossentropy(Loss):

    # forward pass
    def forward(self, y_pred, y_true):

        # num of samples in batch
        samples = len(y_pred)
        print(f"Samples: {samples}")

        # clip data to prevent division by 0
        # clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        print(f"Clipped Predictions: {y_pred_clipped}")

        # probabilities for target values
        # only if categorical labels
        if len(y_true.shape) == 1:
            print("Dealing with - Sparse")
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]
            print(f"Correct Confidences: {correct_confidences}")
        
        # mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            print("Dealing with - One-hot Encoded")
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )
            print(f"Correct Confidences: {correct_confidences}")
        
        # losses
        negative_log_likelihoods = -np.log(correct_confidences)
        print(f"Negative Log Likelihoods: {negative_log_likelihoods}")

        return negative_log_likelihoods

```

```python
loss_function = Loss_CategoricalCrossentropy() # inherits from Loss class
loss = loss_function.calculate(softmax_outputs, class_targets)
print(f"Loss: {loss}")
```


```
Samples: 3
Clipped Predictions: [[0.7  0.1  0.2 ]
 [0.1  0.5  0.4 ]
 [0.02 0.9  0.08]]
Dealing with - One-hot Encoded
Correct Confidences: [0.7 0.5 0.9]
Negative Log Likelihoods: [0.35667494 0.69314718 0.10536052]
Sample Losses:  [0.35667494 0.69314718 0.10536052]
Data Loss: 0.38506088005216804
Loss: 0.38506088005216804
```

# Combining everything up to this point

```python
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# dense layer
class Layer_Dense:

    # layer initialization
    def __init__(self, n_inputs, n_neurons):
        # initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    
    # forward pass
    def forward(self, inputs):
        # calculate output values from inputs, weights, and biases
        self.output = np.dot(inputs, self.weights) + self.biases

# ReLU activation
class Activation_ReLU:
    
        # forward pass
        def forward(self, inputs):
            # calculate output values from inputs
            self.output = np.maximum(0, inputs)

# softmax activation
class Activation_Softmax:
         
        # forward pass
        def forward(self, inputs):
    
            # get unnormalized probabilities
            exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
    
            # normalize them for each sample
            probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    
            self.output = probabilities

# common loss class
class Loss:
         
        # calculates data and regularization losses
        # given model output and ground truth values
        def calculate(self, output, y):
    
            # calculate sample losses
            sample_losses = self.forward(output, y)
            #print("Sample Losses: ", sample_losses)
    
            # calculate mean loss
            data_loss = np.mean(sample_losses)
            #print(f"Data Loss: {data_loss}")
    
            # return loss
            return data_loss
        
# cross-entropy loss
class Loss_CategoricalCrossentropy(Loss):
     
     # forward pass
    def forward(self, y_pred, y_true):
         
        # number of samples in a batch
        samples = len(y_pred)
         
        # clip data to prevent division by 0
        # clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
         
        # probabilities for target values
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
             
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
         
        # losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    
# create dataset
X, y = spiral_data(samples=100, classes=3)

# create dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)

# create ReLU activation (to be used with Dense Layer)
activation1 = Activation_ReLU()

# create second dense layer with 3 input features (as we take output of previous layer here) and 3 output values
dense2 = Layer_Dense(3, 3)

# create softmax activation (to be used with Dense Layer)
activation2 = Activation_Softmax()

# create loss function
loss_function = Loss_CategoricalCrossentropy()

# perform a forward pass of our training data through this layer
dense1.forward(X)

# perform a forward pass through activation function
# takes the output of first dense layer here
activation1.forward(dense1.output)

# perform a forward pass through second Dense Layer
# takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)

# perform a forward pass through activation function
# takes the output of second dense layer here
activation2.forward(dense2.output)

# let's see output of the first few samples:
print(f"Output of Few Samples : \n{activation2.output[:5]}")

# perform a forward pass through activation function
# takes the output of second dense layer here and returns loss
loss = loss_function.calculate(activation2.output, y)

# print loss value
print('Loss:', loss)
```

```
Output of Few Samples : 
[[0.33333334 0.33333334 0.33333334]
 [0.33333316 0.3333332  0.33333364]
 [0.33333287 0.3333329  0.33333418]
 [0.3333326  0.33333263 0.33333477]
 [0.33333233 0.3333324  0.33333528]]
Loss: 1.0986104
Accuracy: 0.34
```

# Accuracy Calculation

```python
import numpy as np

# probabilities of 3 samples
softmax_outputs = np.array([[0.7, 0.2, 0.1],
                                [0.5, 0.1, 0.4],
                                [0.02, 0.9, 0.08]])
# target (ground-truth) labels for 3 samples
class_targets = np.array([0, 1, 1])

# calculate values along second axis (axis of index 1)
predictions = np.argmax(softmax_outputs, axis=1)
print(f"Predictions: {predictions}")
# if targets are one-hot encoded - convert them
if len(class_targets.shape) == 2:
    class_targets = np.argmax(class_targets, axis=1)
    print(f"Class Targets (one hot encode): {class_targets}")
# True evaluates to 1; False to 0
accuracy = np.mean(predictions == class_targets)
print(f"Accuracy: {accuracy}")
```

```
Predictions: [0 0 1]
Accuracy: 0.6666666666666666
```



# Questions
1. What confidence level would 100% sure be in the softmax output?

# YouTube

Neural Networks from Scratch - P.7 Calculating Loss with Categorical Cross-Entropy
https://youtu.be/dEXPMQXoiLc?si=K-aXSe6JBao15ukM



