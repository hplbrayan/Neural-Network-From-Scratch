import numpy as np
from base_activations import BaseActivation

class ReLU(BaseActivation):
    
    def act(self, x):
        return np.where(np.array(x)>0, x, 0)
    
    def dadz(self, x):
        return np.where(np.array(x)>0, 1, 0)