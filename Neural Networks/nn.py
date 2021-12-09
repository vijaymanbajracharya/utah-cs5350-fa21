import numpy as np
from numpy.core.numeric import ones
from numpy.lib.index_tricks import c_
import pandas as pd

EPOCHS = 10
LR_0 = 1e-2
LR_A = 0.75

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
    # train = pd.read_csv("utah-cs5350-fa21/Neural Networks/bank-note/train.csv", names=table)
    # test = pd.read_csv("utah-cs5350-fa21/Neural Networks/bank-note/test.csv", names=table)

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

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class ThreeLayerNN:
    def __init__(self, input_dim, hidden_dim, output_dim, weight_init='rand'):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.params = {}
        self.grads = {}
        self.scores = {}

        if weight_init == 'rand':
            self.params['w1'] = np.random.randn(input_dim, hidden_dim)
            self.params['w2'] = np.random.randn(hidden_dim + 1, hidden_dim)
            self.params['w3'] = np.random.randn(hidden_dim + 1, output_dim)
        
        if weight_init == 'zero':
            # Initialize with zeros instead
            self.params['w1'] = np.zeros((input_dim, hidden_dim))
            self.params['w2'] = np.zeros((hidden_dim + 1, hidden_dim))
            self.params['w3'] = np.zeros((hidden_dim + 1, output_dim))

    def __forward_pass(self, X):
        scores = sigmoid(np.dot(X, self.params['w1']))
        scores = np.c_[np.ones(scores.shape[0]), scores]
        self.scores['z1'] = scores

        scores = sigmoid(np.dot(scores, self.params['w2']))
        scores = np.c_[np.ones(scores.shape[0]), scores]
        self.scores['z2'] = scores

        scores = np.dot(scores, self.params['w3'])
        return scores

    def __loss(self, scores, y):
        loss = np.sum(0.5 * (scores - y)**2)
        return loss

    def __backward_pass(self, scores, X, y):
        # Output Layer
        dL = scores - y
        self.grads['w3'] = dL * self.scores['z2'].T

        # Hidden Layer 2 
        dy = self.params['w3'][1:]
        dsig2 = (self.scores['z2']*(1-self.scores['z2']))[0][1:]
        dz = np.dot(dL, dy.T) * dsig2
        self.grads['w2'] = np.outer(dz, self.scores['z1']).T

        # Hidden Layer 1
        dsig1 = (self.scores['z1']*(1-self.scores['z1']))[0][1:]
        dz = np.dot(dz, (self.params['w2'][1:]).T) * dsig1
        self.grads['w1'] = np.outer(dz, X).T
        

    def backpropagation(self, X, y):
        scores = self.__forward_pass(X)
        loss = self.__loss(scores, y)
        self.__backward_pass(scores, X, y)
        return loss

    def train(self, X, y):

        for i in range(EPOCHS):
            X, y = shuffle(X, y)
            print(f"EPOCH: {i}")

            for index, row in enumerate(X):
                self.learning_rate = LR_0 / (1 + (LR_0/LR_A)*index)
                row = row[np.newaxis,:]
                loss = self.backpropagation(row, y[index])
                self.params['w3'] = self.params['w3'] - self.learning_rate * self.grads['w3']
                self.params['w2'] = self.params['w2'] - self.learning_rate * self.grads['w2']
                self.params['w1'] = self.params['w1'] - self.learning_rate * self.grads['w1']

    def predict(self, X, y):
        predictions = self.__forward_pass(X)
        predictions[predictions < 0] = -1
        predictions[predictions > 0] = 1
        errors = 0
        for i in range(X.shape[0]):
            if predictions[i] != y[i]:
                errors += 1

        return errors / X.shape[0]

if __name__ == "__main__":
    ###################################
    # read data and setup hyperparams #
    ###################################

    # Read data from CSV
    X_train, X_test, y_train, y_test = read_csv()

    hidden_dim_list = [5, 10, 25, 50, 100]

    print(f"Random Normal Initializations")
    for dim in hidden_dim_list:
        # Initialize Model
        model = ThreeLayerNN(4, dim, 1, weight_init='rand')

        # Train the Neural Network
        print(f"Training Neural Network - Hidden {dim}:")
        model.train(X_train, y_train)
        print("\r\n")

        # Find error rates for training and testing sets
        print(f"Predicting Error Rates:")
        train_error = model.predict(X_train, y_train)
        test_error = model.predict(X_test, y_test)
        print(f"Train Error: {train_error * 100}%")
        print(f"Test Error: {test_error * 100}%")
        print(f"\r\n")

    print(f"Zero Initializations")
    for dim in hidden_dim_list:
        # Initialize Model
        model = ThreeLayerNN(4, dim, 1, weight_init='zero')

        # Train the Neural Network
        print(f"Training Neural Network - Hidden {dim}:")
        model.train(X_train, y_train)
        print("\r\n")

        # Find error rates for training and testing sets
        print(f"Predicting Error Rates:")
        train_error = model.predict(X_train, y_train)
        test_error = model.predict(X_test, y_test)
        print(f"Train Error: {train_error * 100}%")
        print(f"Test Error: {test_error * 100}%")
        print(f"\r\n")
