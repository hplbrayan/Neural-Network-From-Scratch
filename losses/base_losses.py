from abc import ABC, abstractmethod

class BaseLosses:
    
    def __call__(self, y_true, y_pred):
        return self.loss(y_true, y_pred)
    
    @abstractmethod
    def loss(self, y_true, y_pred):
        pass

    @abstractmethod
    def dcda(self, y_true, y_pred):
        pass
        
        
   
