import numpy as np

class Losses:
    
    def __init__(self, loss="MSE"):
        
        if loss=='MSE':
            self.loss = self.mse
            self.dcda = self.d_mse
            
            
    #===============================================================================#
    #                                        MSE                                    #
    #===============================================================================#
    def mse(self, y_true, y_pred):
        return np.mean((y_true-y_pred)**2)/2
    
    def d_mse(self, y_true, y_pred):
        return -(y_true-y_pred)/y_true.shape[0]
        
   
