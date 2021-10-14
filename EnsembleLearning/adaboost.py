from os import error
import pandas as pd
import numpy as np
import sys
import time
import copy
#sys.path.append("../")
sys.path.append("utah-cs5350-fa21/")
from DecisionTree.id3 import DecisionTreeClassifier

def read_csv():
    # Loading the Bank dataset
    cols = '''age,
    job,
    marital,
    education,
    default,
    balance,
    housing,
    loan,
    contact,
    day,
    month,
    duration,
    campaign,
    pdays,
    previous,
    poutcome,
    label
    '''
    table = []
    labels = ["yes", "no"]
    attributes = ["age", "job", "marital", "education", "default", 
    "balance", "housing", "loan", "contact", "day", "month", "duration",
    "campaign", "pdays", "previous", "poutcome"]

    for c in cols.split(','):
        if(c.strip()):
            table.append(c.strip())

    #train = pd.read_csv("bank/train.csv", names=table)
    #test = pd.read_csv("bank/test.csv", names=table)
    train = pd.read_csv("utah-cs5350-fa21/DecisionTree/bank/train.csv", names=table)
    test = pd.read_csv("utah-cs5350-fa21/DecisionTree/bank/test.csv", names=table)

    # Binarize the numerical data
    numerical_cols = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
    for atr in numerical_cols:
        train_col = train.loc[:,atr]
        test_col = test.loc[:,atr]
        # Calculate median
        threshold = train_col.median()

        train.loc[train[atr] < threshold, atr] = 0
        train.loc[train[atr] >= threshold, atr] = 1
        test.loc[test[atr] < threshold, atr] = 0
        test.loc[test[atr] >= threshold, atr] = 1

    return train, test, attributes, labels

class Adaboost:
    def __init__(self, no_classifiers=100):
        self.no_classifiers = no_classifiers
        self.classifiers = []
        self.alphas = []
        self.preds = []

    def hypothesize(self, m):
        # initialize weights
        d = np.full(m, 1/m)

        # iterate over the number of weak classifiers we want
        for i in range(self.no_classifiers):
            train, test, attributes, labels = read_csv()
            train["weights"] = d

            # Decision Tree Stump
            h = DecisionTreeClassifier().create_tree(train, train, attributes, labels, gain="info_gain", maxdepth=2, use_weights=1)
            self.classifiers.append(copy.copy(h))

            # Calculate error
            train_pred = np.zeros(m, dtype=np.float64)
            for index, row in train.iterrows():
                pred = DecisionTreeClassifier().predict(row, h)
                if pred == "no":
                    train_pred[index] = -1
                else:
                    train_pred[index] = 1

            target = train["label"].copy().to_numpy()
            target[target == "no"] = -1
            target[target == "yes"] = 1
            target = target.astype(float)

            errors = 0
            for i in range(len(target)):
                if target[i] != train_pred[i]:
                    errors += 1

            train_error = errors / len(train)

            self.preds.append(train_pred)

            # Calculate alpha
            alpha = np.log((1-train_error)/train_error) / 2
            self.alphas.append(copy.copy(alpha))

            # Update weights
            d = d * np.exp(-alpha * target * train_pred)
            d = d / np.sum(d)

        pred = np.zeros(train.shape[0])
        for i in range(self.no_classifiers):
            pred += ada.alphas[i] * ada.preds[i]

        y_pred = np.sign(pred)
        return y_pred

class Bagging:
    def __init__(self, no_classifiers=10):
        self.no_classifiers = no_classifiers
        self.classifiers = []
        self.seed = None

    def fit(self):
        train, test, attributes, labels = read_csv()
        np.random.seed(self.seed)
        for i in range(self.no_classifiers):
            X = train.sample(train.shape[0], replace=True)
            
            tree = DecisionTreeClassifier().create_tree(X, X, attributes, labels, gain="info_gain")

            self.classifiers.append(copy.copy(tree))
                
    def predict(self):
        pass

def predict(row, root):
        value = row[root.value]
        for branches in root.edge:
            if branches.feature_value == value:
                if branches.edge is None:
                    pred = branches.value
                else:
                    pred = predict(row, branches)
        return pred

if __name__ == "__main__":
    ada = Adaboost()
    train, test, attributes, labels = read_csv()

    # Train 
    final_h = ada.hypothesize(train.shape[0])

    # Calculate Trainning Error
    target = train["label"].copy().to_numpy()
    target[target == "no"] = -1
    target[target == "yes"] = 1
    target = target.astype(float)

    errors = 0
    for i in range(len(target)):
        if target[i] != final_h[i]:
            errors += 1

    train_error = errors / len(train)
    print(f"TRAIN ERROR: {train_error}")

    # Calculate Testing Error
    target = test["label"].copy().to_numpy()
    target[target == "no"] = -1
    target[target == "yes"] = 1
    target = target.astype(float)

    test_pred = []
    for cls in range(ada.no_classifiers):
        prediction = np.zeros(test.shape[0], dtype=np.float64)
        for index, row in test.iterrows():
            pred = predict(row, ada.classifiers[cls])
            if pred == "no":
                prediction[index] = -1
            else:
                prediction[index] = 1

        test_pred.append(prediction)

    pred = np.zeros(test.shape[0])
    for i in range(ada.no_classifiers):
        pred += ada.alphas[i] * test_pred[i]

    y_pred = np.sign(pred)

    errors = 0
    for i in range(len(target)):
        if target[i] != y_pred[i]:
            errors += 1

    test_error = errors / len(test)

    print(f"TEST ERROR: {test_error}")

    # bag = Bagging()
    # bag.fit()
    

    



