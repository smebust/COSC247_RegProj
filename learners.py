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
        self.theta = 0

    def fit(self, X, y):
        self.theta = np.linalg.solve(np.dot(np.transpose(X),X), np.dot(np.transpose(X), y))

    def predict(self, X):
        return np.dot(np.transpose(X), self.theta)
        

    
class RidgeRegression(Regressor):
    def __init__(self, lamb):
        self.theta = 0
        self.weight = lamb
        
    def fit(self, X, y):
        I = np.identity(X.shape[0])
        n = X.shape[0]
        a = ((1/n)*np.dot(np.transpose(X), X)) + (self.weight*I)
        b = (1/n)*np.dot(np.transpose(X), y)
        self.theta = np.linalg.solve(a,b)

    def predict(self, X):
        return np.dot(np.transpose(X), self.theta)


class GeneralizedRidgeRegression(Regressor):
    def __init__(self, reg_weights):
        self.theta = 0
        self.lambs = reg_weights
        
    def fit(self, X, y):
        n = X.shape[0]
        a = ((1/n)*np.dot(np.transpose(X), X)) + np.diag(self.lambs)
        b = (1/n)*np.dot(np.transpose(X), y)
        self.theta = np.linalg.solve(a,b)

    def predict(self, X):
        return np.dot(np.transpose(X), self.theta)
    

class DualRidgeRegression(Regressor):
    def __init__(self, lamb, kernel):
        self.a = 0
        self.weight = lamb
        self.kernel = kernel
        
    def fit(self, X, y):
        I = np.identity(X.shape[0])
        K = np.zeros((X.shape[0], X.shape[0]))
        for i in range(K.shape[0]):
            for j in range(K.shape[0]):
                K[i,j] = self.kernel(X[i], X[j])
        x = K + self.weight*I
        self.a = np.linalg.solve(x,y)


    def predict(self, X):
        K = np.zeros((X.shape[0], X.shape[0]))
        for i in range(K.shape[0]):
            for j in range(K.shape[0]):
                K[i,j] = self.kernel(X[i], X[j])
        return np.dot(np.transpose(self.a), K)

    
class AdaptiveLinearRegression(Regressor):
    def __init__(self, kernel): # note kernel used in totally different way
        self.kernel = kernel
        self.X = 0
        self.y = 0

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        toRet = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            K = np.zeros(self.y.shape[0])
            for j in range(self.y.shape[0]):
                K[j] = self.kernel(X[i], self.X[j]) # similarity measure between x and x(i)
            K_mat = np.diag(K)

            a = np.dot(np.transpose(X), K_mat)
            b = np.dot(a, K_mat)
            c = np.dot(b, self.X)
            a1 = np.dot(np.transpose(X), K_mat)
            b1 = np.dot(a1, self.y)
 
            aToDot = np.linalg.solve(c, b1)
            toRet[i] = np.dot(aToDot, self.X[i])

        return toRet



    
