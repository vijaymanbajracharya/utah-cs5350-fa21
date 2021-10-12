import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

COST_LIST = []
WEIGHT_LIST = []
GRAD_LIST = []
STEP_COUNT = 0
LEARNING_RATE = 0.25

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

    y_test = test["label"].copy()
    test = test.iloc[:, 0:-1]
    test = test.to_numpy()
    y_test = y_test.to_numpy()[:,np.newaxis]

    return train, test, y, y_test

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

def batch_descent(train, y, learning_rate=LEARNING_RATE, threshold=10e-6):
    # Initialize weight vector to 0
    global STEP_COUNT
    weights = np.zeros((1, train.shape[1]))
    while True:
        WEIGHT_LIST.append(weights)

        # Calculate LMS cost
        cost = LMScost(train, weights, y)
        COST_LIST.append(cost)

        # Calculate gradient
        grad = LMScost_gradient(train, weights, y)
        GRAD_LIST.append(grad)

        # Update weights
        weights = weights - learning_rate * grad

        # Count steps for plotting
        STEP_COUNT  = STEP_COUNT + 1

        # Error difference magnitude
        diff = np.linalg.norm(weights - WEIGHT_LIST[-1])
        if diff < threshold:
            WEIGHT_LIST.append(weights)
            break

def SGD():
    pass

def predict(test, y, weights):
    cost = LMScost(test, weights, y)
    print(f"Function value (Test): {cost}\r\n")

if __name__ == "__main__":
    train, test, y, y_test = read_csv_basic()
    batch_descent(train, y)

    plt.step(x=np.arange(0, STEP_COUNT, 1),y=COST_LIST)
    plt.show()
    print(f"Wight Vector: {WEIGHT_LIST[-1]}\r\n")
    print(f"Learning Rate: {LEARNING_RATE}\r\n")

    predict(test, y_test, WEIGHT_LIST[-1])

