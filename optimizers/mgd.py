from .base_optimizers import BaseOptimizers
import numpy as np

class MomentumGradientDescent(BaseOptimizers):
    
    def __init__(self, eta=0.1, b=0.9):
        self.optimizer_name = 'gradient_descent_momentum'
        self.eta = eta
        self.b   = b
        self.ncall = 0
            
    def optimizer(self, w, b, grad):

        ws = w
        bs = b

        # after the first call Vw and Vb must save their previous values
        if self.ncall==0:
            self.Vw = {f'Vw_{l}':np.zeros_like(ws['w_' + str(l)]) for l in range(1, len(w.keys())+1)}
            self.Vb = {f'Vb_{l}':np.zeros_like(bs['b_' + str(l)]) for l in range(1, len(w.keys())+1)}           
        
        # update weights
        for l in range(1, len(w.keys())+1):
            self.Vw['Vw_' + str(l)] = self.b*self.Vw['Vw_' + str(l)] + (1-self.b)*grad['dCdW']['dCdW_' + str(l)] 
            self.Vb['Vb_' + str(l)] = self.b*self.Vb['Vb_' + str(l)] + (1-self.b)*grad['dCdb']['dCdb_' + str(l)] 

            ws['w_' + str(l)] -= self.eta*self.Vw['Vw_' + str(l)]
            b['b_' + str(l)]  -= self.eta*self.Vb['Vb_' + str(l)]
                
            self.ncall +=1
            
        return ws, bs
        
