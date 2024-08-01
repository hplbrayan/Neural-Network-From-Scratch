import numpy as np
from .base_activations import BaseActivation

class Tanh(BaseActivation):
    def act(self, x):
        return np.tanh(x)
    
    def dadz(self, x):
        return 1 - np.tanh(x)**2
