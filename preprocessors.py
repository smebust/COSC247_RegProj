import numpy as np

debug = False

def prepend_1s(X):
    toRet = np.insert(X, 0, 1, axis=1)
    return toRet


def poly_lift(X, degree):
    X1 = []
    for i in range(degree):
        toAp = []
        for k in range(X.shape[0]):
            toAp.append(X[k]**i)
        X1.append(toAp)

    toRet= np.array(X1)
    toRet = np.transpose(toRet)

    return toRet


def standardize(X):
    Xt = X
    for j in range(Xt.shape[1]):
        thisMax = np.max(X[:,j])
        if(debug):
            print("thisMax")
            print(thisMax)
        thisMin = np.min(X[:,j])
        if(debug):
            print("thisMin")
            print(thisMin)
        thisRange = thisMax-thisMin
        if(debug):
            print("thisRange")
            print(thisRange)
            print(" ")
        for i in range(Xt[:,j].shape[0]):
            if(debug):
                print(" ")
                print("i:")
                print(Xt[int(i),j])
            if(Xt[int(i),j]==thisMax):
                if(debug):
                    print("Entered thisMax")
                Xt[int(i),j] = 1
                if(debug):
                    print("i in thismax")
                    print(Xt[int(i),j])
            elif(Xt[int(i),j]==thisMin):
                if(debug):
                    print("Entered thisMin")
                Xt[int(i),j] = 0
                if(debug):
                    print("i in thismin")
                    print(Xt[int(i),j])
            else:
                Xt[int(i),j] = (Xt[int(i),j]-thisMin)/thisRange
                if(debug):
                    print("i in else")
                    print(Xt[int(i),j])    
    if(debug):
        print(Xt)
    return Xt


