"""import sys
import os

# Current dir
current_dir = os.path.dirname(os.path.abspath('__file__'))
project_root = os.path.abspath(os.path.join(current_dir, '..'))

# add current dir to sys.path
if project_root not in sys.path: sys.path.append(project_root)"""

########################################################################
########################################################################
import activations.relu as relu

relu.ReLU()

