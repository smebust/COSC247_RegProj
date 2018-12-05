import numpy as np

# A decorator -- don't worry if you don't understand this.
# It just makes it so that each loss function you implement automatically checks that arguments have the same number of elements
# Function provided by prof. Scott Alfeld
def loss_fun(fun):
    def toR(y_true, y_preds):
        n,  = y_true.shape
        npreds, = y_preds.shape
        assert n == npreds, "There must be as many predictions as there are true values"
        return fun(y_true, y_preds)
    return toR

# Function provided by prof. Scott Alfeld
@loss_fun
def zero_one(y_true, y_preds):
    n, = y_true.shape
    return np.sum([1 for yt, yp in zip(y_true, y_preds) if yt == yp ])/n

# Rest of method headers provided by prof. Scott Alfeld, body by Sean Mebust
@loss_fun
def MSE(y_true, y_preds):
    op = []
    n, = y_true.shape
    for i in range(0,n):
        op.append((y_preds[i] - y_true[i])**2)
    toRet = np.array((sum(op))/n)
    return toRet

@loss_fun
def MAD(y_true, y_preds):
    op = []
    n, = y_true.shape
    for i in range(0,n):
        op.append(np.sqrt((y_preds[i] - y_true[i])**2))
    toRet = np.array((sum(op))/n)
    return toRet

def cross_validation(X, y, reg, evaler, num_folds = 10):
    d = X.shape[1]
    n = y.shape[0]
    toAv = 0
    for k in range(num_folds):
        ykL = [] #add point when mod(index,numfolds) is k 
        yelseL = [] #rest of result values
        for i in range(y.shape[0]):
            if (i%num_folds == k):
                ykL.append(y[i])
            else:
                yelseL.append(y[i])
        yk = np.array(ykL)
        yelse = np.array(yelseL)

        xk = np.zeros((yk.shape[0], X.shape[1]))
        xelse = np.zeros((yelse.shape[0], X.shape[1]))
        for j in range(d):
            xkL = []
            xelseL = []
            for i in range(int(y.shape[0])):
                if (i%num_folds == k):
                    xkL.append(X[i,j])
                else:
                    xelseL.append(X[i,j])
            if(xkL):
                xk[:,j] = np.array(xkL)          
            if(xelseL):          
                xelse[:,j] = np.array(xelseL)
                
        """
        print("X")
        print(X)
        print("y")
        print(y)

        print("yk")
        print(yk)
        print("yelse")
        print(yelse)
        print("xk")
        print(xk)
        print("xelse")
        print(xelse)
        """

        r = reg
        r.fit(xelse, yelse) #fits/creates a model for test (left out) points
        yOut = r.predict(xk) #predicts values for output of remaining points
        res = evaler(yk, yOut) #evaluates loss on model against true output
        toAv += res #adds score to sum to be averaged

    return toAv/num_folds


    
