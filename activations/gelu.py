import numpy as np
from scipy.special import erf
from .base_activations import BaseActivation

class GELU(BaseActivation):
    def act(self, x):
        return 0.5*(1 + erf(x/np.sqrt(2)))
    
    def dadz(self, x):
        return 0.5*(1 + erf(x/np.sqrt(2))) + x*np.exp(-x**2/2)/np.sqrt(2*np.pi)
