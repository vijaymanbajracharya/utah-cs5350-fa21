import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import Bounds
import warnings

EPOCHS = 100
LR_0 = 1e-2
LR_A = 1

def read_csv():
    # Loading the Concrete dataset
    cols = '''
    variance,
    skewness,
    curtosis,
    entropy,
    label
    '''
    table = []

    for c in cols.split(','):
        if(c.strip()):
            table.append(c.strip())

    train = pd.read_csv("bank-note/train.csv", names=table)
    test = pd.read_csv("bank-note/test.csv", names=table)
    #train = pd.read_csv("utah-cs5350-fa21/SVM/bank-note/train.csv", names=table)
    #test = pd.read_csv("utah-cs5350-fa21/SVM/bank-note/test.csv", names=table)

    y = train["label"].copy()
    train = train.iloc[:, 0:-1]
    train = train.to_numpy()
    y = y.to_numpy()[:,np.newaxis]
    y[y == 0] = -1

    y_test = test["label"].copy()
    test = test.iloc[:, 0:-1]
    test = test.to_numpy()
    y_test = y_test.to_numpy()[:,np.newaxis]
    y_test[y_test == 0] = -1

    return train, test, y, y_test

def shuffle(X, y):
    shuffler = np.random.permutation(len(X))
    X_shuffled = X[shuffler]
    y_shuffled = y[shuffler]
    return X_shuffled, y_shuffled

def gaussian_kernel(xi, xj, gamma=0.1):
    kernel = np.exp((-np.linalg.norm(xi-xj)**2) / gamma)
    return kernel

class svm:
    def __init__(self, C=100/873, lr_schedule=0):
        self.C = C
        self.weights = None
        self.learning_rate = LR_0
        self.lr_schedule = lr_schedule

    def fit(self, X, y):
        n = X.shape[0]

        # Default weights and fold bias
        weights_0 = np.zeros((X.shape[1], 1))
        weights = np.r_[weights_0, [[0]]]
        X = np.c_[np.ones(X.shape[0]), X]
        
        for i in range(EPOCHS):
            X, y = shuffle(X, y)

            for index, row in enumerate(X):
                if self.lr_schedule == 0 and index > 0:
                    self.learning_rate = LR_0 / (1 + (LR_0/LR_A)*index)

                if self.lr_schedule == 1 and index > 0:
                    self.learning_rate = LR_0 / (1+index)

                row = row[:,np.newaxis]
                functional_margin = y[index][0]*np.dot(weights.T, row)
                if functional_margin <= 1:
                    weights = weights - self.learning_rate * np.r_[weights_0, [[0]]] + self.learning_rate * self.C * n * (y[index] * row)
                else:
                    weights_0 = (1-self.learning_rate) * weights_0

        self.weights = weights

    def predict(self, X, y):
        bias = self.weights[0]
        predictions = np.sign(np.dot(X, self.weights[1:]) + bias)
        errors = 0
        for i in range(X.shape[0]):
            if predictions[i] != y[i]:
                errors += 1

        return errors / X.shape[0]

class svm_dual:
    def __init__(self, C=100/873, use_kernel=0, gamma=0.1):
        self.C = C
        self.alphas = None
        self.G = None
        self.learning_rate = LR_0
        self.weights = None
        self.bias = None
        self.use_kernel = use_kernel
        self.gamma = gamma

    def obj_func(self, alphas):
        return 0.5 * np.dot(alphas.T, np.dot(self.G, alphas)) - np.sum(alphas)

    def fit(self, X, y):
        if self.use_kernel == 0:
            self.G = np.dot(X, X.T)*np.dot(y, y.T)
        else:
            self.G = np.zeros((X.shape[0],X.shape[0]))
            for i in range(X.shape[0]):
                for j in range(X.shape[0]):
                    self.G[i,j] = gaussian_kernel(X[i], X[j])

            self.G = self.G * np.dot(y, y.T)
       
        bnds = Bounds(0, self.C)
        cons = {'type': 'eq', 'fun': lambda alphas: np.dot(alphas, y)}
        alpha_0 = np.zeros(X.shape[0])

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            sol = minimize(self.obj_func, alpha_0, method='SLSQP', bounds=bnds, constraints=cons)
            
        self.alphas = sol.x

    def recover_params(self, X, y, alphas):
        self.weights = np.dot(X.T * alphas, y)
        if self.use_kernel == 0:
            self.bias = np.mean(y - np.dot(X, self.weights))
        else:
            kernel = np.zeros((X.shape[0], X.shape[0]))
            for i in range(X.shape[0]):
                for j in range(X.shape[0]):
                    kernel[i,j] = gaussian_kernel(X[i], X[j], self.gamma)
            self.bias = np.mean(y - alphas*y*kernel)
        
    def predict(self, X, y):
        if self.use_kernel == 0:
            predictions = np.sign(np.dot(X, self.weights) + self.bias)
        else:
            predictions = np.zeros((X.shape[0]))
            for i in range(X.shape[0]):
                kernel = np.zeros((1, X.shape[0]))
                for j in range(X.shape[0]):
                    kernel[0, j] = gaussian_kernel(X[i], X[j], self.gamma)
                predictions[i] = np.sum(self.alphas[i] * y[i] * np.sum(kernel))
            predictions = np.sign(predictions + self.bias)

        errors = 0
        for i in range(X.shape[0]):
            if predictions[i] != y[i]:
                errors += 1

        return errors / X.shape[0]


if __name__ == "__main__":

    ###################################
    # read data and setup hyperparams #
    ###################################
    train, test, y, y_test = read_csv()
    C = [100/873, 500/873, 700/873]

    ##############
    # svm primal #
    ##############
    print("SVM Primal Domain:\r\n")
    print("Learning Rate Schedule 0\r\n")
    for param in C:
        model = svm(C=param, lr_schedule=0)
        model.fit(train, y)
        print(f"For C: {param}")
        print(f"Learned Weights: {model.weights.T}")
        print(f"Train Error: {round(model.predict(train, y) * 100, 3)}%")
        print(f"Test Error: {round(model.predict(test, y_test) * 100, 3)}%")
        print("\r\n")

    print("Learning Rate Schedule 1\r\n")
    for param in C:
        model = svm(C=param, lr_schedule=1)
        model.fit(train, y)
        print(f"For C: {param}")
        print(f"Learned Weights: {model.weights.T}")
        print(f"Train Error: {round(model.predict(train, y) * 100, 3)}%")
        print(f"Test Error: {round(model.predict(test, y_test) * 100, 3)}%")
        print("\r\n")

    ############
    # svm dual #
    ############
    print("SVM Dual Domain: This will take a few minutes to complete\r\n")

    for param in C:
        model = svm_dual(C=param, use_kernel=0)
        model.fit(train, y)
        model.recover_params(train, y, model.alphas)
        print(f"For C: {param}")
        print(f"Recovered Weights: {model.weights.T}")
        print(f"Recovered Bias: {model.bias}")
        print(f"Train Error: {round(model.predict(train, y) * 100, 3)}%")
        print(f"Test Error: {round(model.predict(test, y_test) * 100, 3)}%")
        print("\r\n")

    ###############################
    # svm dual w/ Gaussian Kernel #
    ############################### 
    G = [0.1, 0.5, 1, 5, 100]
    print("SVM Dual Domain w/ Gaussian Kernel: This will take a few minutes to complete\r\n")
    for param in C:
        for gamma in G:
            model = svm_dual(C=param, use_kernel=1, gamma=gamma)
            model.fit(train, y)
            model.recover_params(train, y, model.alphas)
            print(f"For C: {param} and gamma: {gamma}")
            print(f"Recovered Weights: {model.weights.T}")
            print(f"Recovered Bias: {model.bias}")
            print(f"Train Error: {round(model.predict(train, y) * 100, 3)}%")
            print(f"Test Error: {round(model.predict(test, y_test) * 100, 3)}%")
            print("\r\n")


