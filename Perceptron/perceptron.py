import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

LEARNING_RATE = 1e-2
EPOCHS = 10

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

def calculate_error(X, y, weights):
    predictions = np.sign(np.dot(X, weights))
    errors = 0
    for i in range(X.shape[0]):
        if predictions[i] != y[i]:
            errors += 1

    return errors / X.shape[0]

class Perceptron:
    def __init__(self):
        self.weights_list = []

    def fit(self, train, y):
        weights = np.zeros((train.shape[1], 1))
        self.weights_list.append(weights)

        for i in range(EPOCHS):
            index = 0
            for row in train:
                row = row[:,np.newaxis]
                prediction = np.sign(np.dot(weights.T, row))
                if y[index] != prediction:
                    weights = weights + LEARNING_RATE * (y[index] * row)
                    self.weights_list.append(weights)

                index += 1

if __name__ == "__main__":
    train, test, y, y_test = read_csv()
    p = Perceptron()
    p.fit(train, y)
    learned_weights = p.weights_list[-1]
    print(f"Learned Weight Vector: {learned_weights.T}")
    test_error = calculate_error(test, y_test, learned_weights)
    print(f"Error: {test_error*100}%")

