from .base_optimizers import BaseOptimizers

class GradientDescent(BaseOptimizers):
        
        def __init__(self, eta=0.1):
            self.optimizer_name = 'gradient_descent'
            self.eta = eta
            
        def optimizer(self, w, b, grad):

            ws = w
            bs = b

            for l in range(1, len(w.keys())+1):
                ws['w_' + str(l)] -= self.eta*grad['dCdW']['dCdW_' + str(l)] 
                b['b_' + str(l)]  -= self.eta*grad['dCdb']['dCdb_' + str(l)]

            return ws, bs
        