from .base_losses import BaseLosses
import numpy as np

class BCE(BaseLosses):           
   
    def loss(self, y_true, y_pred):
        return -np.mean(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred))
    
    def dcda(self, y_true, y_pred):
        return ((1-y_true)/(1-y_pred) - y_true/y_pred)/y_true.shape[0]
    
        
   
