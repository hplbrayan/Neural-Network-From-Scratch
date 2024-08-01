from .base_losses import BaseLosses
import numpy as np

class MSE(BaseLosses):           
   
    def loss(self, y_true, y_pred):
        return np.mean((y_true-y_pred)**2)/2
    
    def dcda(self, y_true, y_pred):
        return -(y_true-y_pred)/y_true.shape[0]
        
   
