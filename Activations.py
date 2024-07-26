import numpy as np

class Activations:
    
    def __init__(self, act='relu'):
        
        if act=='relu':
            self.act = self.relu
            self.dadz = self.d_relu
            
        if act=='identity':
            self.act = self.identity
            self.dadz = self.d_identity
            
            
    #===============================================================================#
    #                                        ReLU                                   #
    #===============================================================================#
    def relu(self, x):
        return np.where(np.array(x)>0, x, 0)
    
    def d_relu(self, x):
        return np.where(np.array(x)>0, 1, 0)
    
    #===============================================================================#
    #                                      Identity                                 #
    #===============================================================================#
    def identity(self, x):
        return np.where(np.array(x)==np.array(x), x, 0)
    
    def d_identity(self, x):
        return np.where(np.array(x)==np.array(x), 1, 0)
   
