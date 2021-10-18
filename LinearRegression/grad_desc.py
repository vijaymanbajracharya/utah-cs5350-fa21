import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

COST_LIST = []
WEIGHT_LIST = []
GRAD_LIST = []
STEP_COUNT = 0
LEARNING_RATE = 1e-2

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

    train = pd.read_csv("utah-cs5350-fa21/LinearRegression/debug_data.csv", names=table)
    test = pd.read_csv("utah-cs5350-fa21/LinearRegression/debug_data.csv", names=table)
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
    return grad 

def LMScost_SGD_gradient(train, weights, y):
    m = train.shape[0]
    scores = np.dot(train, weights.T)
    error = y - scores
    grad = -1 * error * train
    return grad 

def batch_descent(train, y, learning_rate=LEARNING_RATE, threshold=10e-6):
    # Initialize weight vector to 0
    global STEP_COUNT
    global COST_LIST
    global WEIGHT_LIST
    global GRAD_LIST
    
    COST_LIST = []
    WEIGHT_LIST = []
    GRAD_LIST = []
    STEP_COUNT = 0
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

def SGD(train, y, learning_rate=LEARNING_RATE, threshold=10e-6):
    # Initialize weight vector to 0
    global STEP_COUNT
    global COST_LIST
    global WEIGHT_LIST
    global GRAD_LIST

    COST_LIST = []
    WEIGHT_LIST = []
    GRAD_LIST = []
    STEP_COUNT = 0
    weights = np.zeros((1, train.shape[1]))
    while True:
        WEIGHT_LIST.append(weights)

        # Calculate LMS cost
        cost = LMScost(train, weights, y)
        COST_LIST.append(cost)

        # Random sample
        index = np.random.randint(train.shape[0])

        # Calculate gradient
        grad = LMScost_SGD_gradient(train[index], weights, y[index])
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


def predict(test, y, weights):
    cost = LMScost(test, weights, y)
    print(f"Function value (Test): {cost}\r\n")

if __name__ == "__main__":
    train, test, y, y_test = read_csv()

    batch_descent(train, y)

    plt.step(x=np.arange(0, STEP_COUNT, 1),y=COST_LIST)
    plt.xlabel("Update Step")
    plt.ylabel("Cost Function Value")
    plt.title("Batch Descent")
    print(f"BD Weight Vector: {WEIGHT_LIST[-1]}")
    print(f"BD Learning Rate: {LEARNING_RATE}")
    predict(test, y_test, WEIGHT_LIST[-1])

    #plt.show()

    SGD(train, y)
    plt.step(x=np.arange(0, STEP_COUNT, 1),y=COST_LIST)
    plt.xlabel("Update Step")
    plt.ylabel("Cost Function Value")
    plt.title("Stochastic Gradient Descent")
    print(f"SGD Weight Vector: {WEIGHT_LIST[-1]}")
    print(f"SGD Learning Rate: {LEARNING_RATE}")
    predict(test, y_test, WEIGHT_LIST[-1])

    #plt.show()

    #w = np.dot(np.linalg.inv(np.dot(test.T, test)), np.dot(test.T, y_test))   
    #print(w.T)

