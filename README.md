# DeepNumpy
This repository contains code to implement Machine/Deep Learning functionalities using Numpy Python

Layers, Activation functions, Forward and Back Propagations are implemented in deepnumpy and made it available as a package

A sample script file run_example.py is also provided which will demonstrate the usage of this code as a library.

```
python run_example.py
```

## Layers:

As of now, only Linear layer is implemented with forward and backward propagation
More layers will be added in future

## Activation Functions:

Relu and Sigmoid functions are implemented. Others will be added in future versions

## Loss Functions:

BinaryCrossEntropy is implemented for classification and MeanSquaredError is implemented for regression.
Other loss functions will be added in future versions

## Metrics:

Only Classification performance metrics are implemented.
Accuracy, Precision, Recall, Confusion Matrix, ROC AUC curve shall be calculated.

## Training Mechanism:

As of now, Constant learning Rate and Gradient Descent is implemented

## Logging:
Logging is implemented. It should configued in the main file to get all the details. Training, Testing performance metrics and all other details will be available in the log file.

## Plots:
Train Test Loss curve and the ROC AUC curve shall be plotted and is saved in output folder.

## Early Stopping
Early Stopping mechanism is available for which the monitoring signal, Inc or Dec, and Patience epochs should be specified. 
For Example: 
```
EARLY_STOPPING_DICT = {'val_loss': ('inc', 5)}
```
In this, Monitoring shall be done for valiation loss value. If it is continuosly higher than its previous epoch's value for atleast 5 epochs, then Early Stopping shall occur
