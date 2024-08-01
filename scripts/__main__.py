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
import activations.relu as relu
import activations.tanh as tanh

# Example usage
activation_function = relu.ReLU()
output = activation_function([1, -1, 5, 2])
print(output)
print(tanh.Tanh()([1,2,3,4,5,6]))

