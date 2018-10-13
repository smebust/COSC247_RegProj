import numpy as np
from abc import ABC, abstractmethod

class Regressor(ABC):
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass


class NotYetTrainedException(Exception):
    """Learner must be trained (fit) before it can predict points."""
    pass


def simple_kernel(x1, x2):
    return (np.dot(x1, x2) + 1)**2


class ToyRegressor(Regressor):
    def __init__(self):
        self.mean = None
        
    def fit(self, X, y):
        self.mean = np.average(y)
        

    def predict(self, X):
        if self.mean is not None:
            return np.array([self.mean for _ in X])
        else:
            raise NotYetTrainedException
        pass
    
    
class OLS(Regressor):
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    
class RidgeRegression(Regressor):
    def __init__(self, lamb):
        pass
        
    def fit(self, X, y):
        pass

    def predict(self, X):
        pass


class GeneralizedRidgeRegression(Regressor):
    def __init__(self, reg_weights):
        pass
        
    def fit(self, X, y):
        pass

    def predict(self, X):
        pass
    

class DualRidgeRegression(Regressor):
    def __init__(self, lamb, kernel):
        pass
        
    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    
class AdaptiveLinearRegression(Regressor):
    def __init__(self, kernel): # note kernel used in totally different way
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass
    
