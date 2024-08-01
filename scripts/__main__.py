"""import sys
import os

# Current dir
current_dir = os.path.dirname(os.path.abspath(__file__))

# Project root dir
project_root = os.path.abspath(os.path.join(current_dir, '..'))

# Add project root dir to sys.path
if project_root not in sys.path:
    sys.path.append(project_root)"""

# Import modules
import numpy as np
import activations.relu as relu
import activations.tanh as tanh
import neuralnetwork.neuralNetwork as nnw
import losses.mse as mse
import optimizers.gd as gd

# Example usage
activation_function = relu.ReLU()
output = activation_function([1, -1, 5, 2])
print(output)
print(tanh.Tanh()([1,2,3,4,5,6]))
print(mse.MSE().loss(y_true=np.array([1,2,3]), y_pred=np.array([1.2, 2.1, 3.4])))

