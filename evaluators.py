import numpy as np

# A decorator -- don't worry if you don't understand this.
# It just makes it so that each loss function you implement automatically checks that arguments have the same number of elements
def loss_fun(fun):
    def toR(y_true, y_preds):
        n,  = y_true.shape
        npreds, = y_preds.shape
        assert n == npreds, "There must be as many predictions as there are true values"
        return fun(y_true, y_preds)
    return toR

@loss_fun
def zero_one(y_true, y_preds):
    n, = y_true.shape
    return np.sum([1 for yt, yp in zip(y_true, y_preds) if yt == yp ])/n

@loss_fun
def MSE(y_true, y_preds):
    pass

@loss_fun
def MAD(y_true, y_preds):
    pass

def cross_validation(X, y, reg, evaler, num_folds = 10):
    pass
