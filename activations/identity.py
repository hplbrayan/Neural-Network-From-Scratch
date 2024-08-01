import numpy as np
from .base_activations import BaseActivation

class Identity(BaseActivation):
    def act(self, x):
        return np.where(np.array(x)==np.array(x), x, 0)
    
    def dadz(self, x):
        return np.where(np.array(x)==np.array(x), 1, 0)
