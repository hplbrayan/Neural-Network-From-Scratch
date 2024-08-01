from abc import ABC, abstractmethod
from typing import Any

class BaseOptimizers(ABC):
    def __call__(self, w, b, grad ):
        return self.optimizer(self, w, b, grad)
    
    @abstractmethod
    def optimizer(self, w, b, grad):
        pass