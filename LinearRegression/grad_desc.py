import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

cost_list = []
weight_list = []
grad_list = []

def read_csv_basic():
    # Loading the Concrete dataset
    cols = '''
    x1,
    x2,
    x3,
    label
    '''
    table = []

    for c in cols.split(','):
        if(c.strip()):
            table.append(c.strip())

    train = pd.read_csv("debug_data.csv", names=table)
    test = pd.read_csv("debug_data.csv", names=table)
    y = train["label"].copy()
    train = train.iloc[:, 0:-1]
    train = train.to_numpy()
    y = y.to_numpy()[:,np.newaxis]

    return train, test, y

def read_csv():
    # Loading the Concrete dataset
    cols = '''
    Cement,
    Slag,
    Fly ash,
    Water,
    SP,
    Coarse Aggr,
    Fine Aggr,
    label
    '''
    table = []

    for c in cols.split(','):
        if(c.strip()):
            table.append(c.strip())

    train = pd.read_csv("concrete/train.csv", names=table)
    test = pd.read_csv("concrete/test.csv", names=table)

    y = train["label"].copy()
    train = train.iloc[:, 0:-1]
    train = train.to_numpy()
    y = y.to_numpy()[:,np.newaxis]

    y_test = test["label"].copy()
    test = test.iloc[:, 0:-1]
    test = test.to_numpy()
    y_test = y_test.to_numpy()[:,np.newaxis]

    return train, test, y, y_test

def LMScost(train, weights, y):
    m = train.shape[0]
    scores = np.dot(train, weights.T).reshape(m, 1)
    squared_err = (y - scores)**2
    sum_err = np.sum(squared_err, axis=0)/2
    return sum_err[0]

def LMScost_gradient(train, weights, y):
    m = train.shape[0]
    scores = np.dot(train, weights.T).reshape(m, 1)
    error = y - scores
    grad = -1 * np.dot(error.T, train) 
    return grad / m

# def LMScost(train, weights, y):
#     sum = 0.0
#     num_train = train.shape[0]
#     for i in range(num_train):
#         score = np.dot(train[i], weights.T)
#         error = y[i] - score
#         sqr_error = np.square(error)
#         sum = sum + sqr_error[0]
    
#     return sum/2

# def LMScost_gradient(train, weights, y):
#     grad = np.zeros(train.shape[1])[np.newaxis, :]
#     num_train = train.shape[0]
#     for i in range(num_train):
#         score = np.dot(train[i], weights.T)
#         error = y[i] - score
#         grad = grad + train[i] * error

#     grad = -1 * grad
#     return grad / train.shape[1]

def batch_descent(train, y, learning_rate=0.25, threshold=10e-6):
    # Initialize weight vector to 0
    weights = np.zeros((1, train.shape[1]))
    while True:
        weight_list.append(weights)

        # Calculate LMS cost
        cost = LMScost(train, weights, y)
        cost_list.append(cost)

        # Calculate gradient
        grad = LMScost_gradient(train, weights, y)
        grad_list.append(grad)

        # Update weights
        weights = weights - learning_rate * grad

        # Error difference magnitude
        diff = np.linalg.norm(weights - weight_list[-1])
        if diff < threshold:
            weight_list.append(weights)
            break

def predict(test, y, weights):
    pass

if __name__ == "__main__":
    train, test, y, y_test = read_csv()
    batch_descent(train, y)
    predict(test, y_test, weight_list[-1])