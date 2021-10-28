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

def calculate_error_std(X, y, weights):
    predictions = np.sign(np.dot(X, weights))
    errors = 0
    for i in range(X.shape[0]):
        if predictions[i] != y[i]:
            errors += 1

    return errors / X.shape[0]

def calculate_error_voted(X, y, weights):
    sum = np.zeros((X.shape[0],1))
    for tuple in weights:
        sum = sum + tuple[1] * np.sign(np.dot(X, tuple[0]))

    predictions = np.sign(sum)
    errors = 0
    for i in range(X.shape[0]):
        if predictions[i] != y[i]:
            errors += 1

    return errors / X.shape[0]

def calculate_error_average(X, y, average):
    predictions = np.sign(np.dot(X, average))
    errors = 0
    for i in range(X.shape[0]):
        if predictions[i] != y[i]:
            errors += 1

    return errors / X.shape[0]

def shuffle(X, y):
    shuffler = np.random.permutation(len(X))
    X_shuffled = X[shuffler]
    y_shuffled = y[shuffler]
    return X_shuffled, y_shuffled

class Perceptron:
    def __init__(self):
        self.weights_list = []

    def fit(self, train, y):
        weights = np.zeros((train.shape[1], 1))
        self.weights_list.append(weights)

        for i in range(EPOCHS):
            index = 0
            train_shuffled, y_shuffled = shuffle(train, y)
            for row in train_shuffled:
                row = row[:,np.newaxis]
                prediction = np.sign(np.dot(weights.T, row))
                if y_shuffled[index] != prediction:
                    weights = weights + LEARNING_RATE * (y_shuffled[index] * row)
                    self.weights_list.append(weights)

                index += 1

class VotedPerceptron:
    def __init__(self):
        self.weights_list = []

    def fit(self, train, y):
        weights = np.zeros((train.shape[1], 1))
        self.weights_list.append((weights, 1))

        for i in range(EPOCHS):
            index = 0
            train_shuffled, y_shuffled = shuffle(train, y)
            for row in train_shuffled:
                row = row[:,np.newaxis]
                prediction = np.sign(np.dot(weights.T, row))
                if y_shuffled[index] != prediction:
                    weights = weights + LEARNING_RATE * (y_shuffled[index] * row)
                    self.weights_list.append((weights, 1))
                else:
                    self.weights_list[-1] = (weights, self.weights_list[-1][1] + 1)

                index += 1

class AveragePerceptron:
    def __init__(self):
        self.average = None

    def fit(self, train, y):
        weights = np.zeros((train.shape[1], 1))
        self.average = np.zeros((train.shape[1], 1))

        for i in range(EPOCHS):
            index = 0
            train_shuffled, y_shuffled = shuffle(train, y)
            for row in train_shuffled:
                row = row[:,np.newaxis]
                prediction = np.sign(np.dot(weights.T, row))
                if y_shuffled[index] != prediction:
                    weights = weights + LEARNING_RATE * (y_shuffled[index] * row)

                self.average += weights
                index += 1

if __name__ == "__main__":
    # Standard Perceptron
    print("Standard Perceptron Algorithm")
    train, test, y, y_test = read_csv()
    p = Perceptron()
    p.fit(train, y)
    learned_weights = p.weights_list[-1]
    print(f"Learned Weight Vector: {learned_weights.T}")
    test_error = calculate_error_std(test, y_test, learned_weights)
    print(f"Test Error: {test_error*100}%")
    print("\r\n")

    # Voted Perceptron
    print("Voted Perceptron Algorithm")
    train, test, y, y_test = read_csv()
    vp = VotedPerceptron()
    vp.fit(train, y)
    test_error = calculate_error_voted(test, y_test, vp.weights_list)
    print(f"Test Error: {test_error*100}%")
    print("\r\n")

    # Average Perceptron
    print("Average Perceptron Algorithm")
    train, test, y, y_test = read_csv()
    ap = AveragePerceptron()
    ap.fit(train, y)
    test_error = calculate_error_average(test, y_test, ap.average)
    print(f"Test Error: {test_error*100}%")



