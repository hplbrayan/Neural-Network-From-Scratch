import numpy as np
from .base_activations import BaseActivation

class Sigmoid(BaseActivation):
    def act(self, x):
        return 1/(1 + np.exp(-x))
    
    def dadz(self, x):
        return self.act(x)*(1 - self.act(x))