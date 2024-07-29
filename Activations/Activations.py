import numpy as np

class Activations:
    
    def __init__(self, act='relu'):
        
        if act=='relu':
            self.act = self.relu
            self.dadz = self.d_relu
            
        if act=='lrelu':
            self.act = self.lrelu
            self.dadz = self.d_lrelu
            self.a = 0.1
            
        if act=='elu':
            self.act = self.elu
            self.dadz = self.d_elu
            self.a = 1
        
        if act=='gelu':
            from scipy.special import erf
            self.act = self.gelu
            self.dadz = self.d_gelu
            
        if act=='identity':
            self.act = self.identity
            self.dadz = self.d_identity
            
        if act=='sigmoid':
            self.act = self.sigmoid
            self.dadz = self.d_sigmoid
            
        if act=='tanh':
            self.act = self.tanh
            self.dadz = self.d_tanh
            
            
    #===============================================================================#
    #                                      Identity                                 #
    #===============================================================================#
    def identity(self, x):
        return np.where(np.array(x)==np.array(x), x, 0)
    
    def d_identity(self, x):
        return np.where(np.array(x)==np.array(x), 1, 0)
    
            
    #===============================================================================#
    #                                        ReLU                                   #
    #===============================================================================#
    def relu(self, x):
        return np.where(np.array(x)>0, x, 0)
    
    def d_relu(self, x):
        return np.where(np.array(x)>0, 1, 0)
    
    
    #===============================================================================#
    #                                    Leaky ReLU                                 #
    #===============================================================================#
    def lrelu(self, x, a=self.a):
        return np.where(np.array(x)>0, x, a*x)
    
    def d_lrelu(self, x, a=self.a):
        return np.where(np.array(x)>0, 1, a)
    
    
    #===============================================================================#
    #                                        ELU                                    #
    #===============================================================================#
    def elu(self, x, a=self.a):
        return np.where(np.array(x)>0, x, a*(np.exp(x) - 1))
    
    def d_elu(self, x, a=self.a):
        return np.where(np.array(x)>0, 1, a*np.exp(x))
    
    
    #===============================================================================#
    #                                        GELU                                   #
    #===============================================================================#
    def gelu(x):
        return 0.5*(1 + erf(x/np.sqrt(2)))
        
    def d_gelu(x):
        return 0.5*(1 + erf(x/np.sqrt(2))) + x*np.exp(-x**2/2)/np.sqrt(2*np.pi)
    
    
    #===============================================================================#
    #                                      Sigmoid                                 #
    #===============================================================================#
    def sigmoid(self, x):
        return 1/(1 + np.exp(x))
    
    def d_sigmoid(self, x):
        return self.sigmoid(x)*(1 - self.sigmoid(x))
    
    
    #===============================================================================#
    #                                        tanh                                   #
    #===============================================================================#
    def tanh(self, x):
        return np.tanh(x)
    
    def d_tanh(self, x):
        return 1 - np.tanh(x)**2
    

