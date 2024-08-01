from abc import ABC, abstractmethod

class BaseActivation(ABC):
    
    def __call__(self, x):
        return self.act(x)
    
    @abstractmethod
    def act(self, x):
        pass
    
    @abstractmethod
    def dadz(self, x):
        pass
