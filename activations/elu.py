import numpy as np
from .base_activations import BaseActivation

class ELU(BaseActivation):
    def __init__(self, a=1) -> None:
        super().__init__()
        self.a = a

    def act(self, x):
        return np.where(np.array(x)>0, x, self.a*(np.exp(x) - 1))
    
    def dadz(self, x):
        return np.where(np.array(x)>0, 1, self.a*np.exp(x))

