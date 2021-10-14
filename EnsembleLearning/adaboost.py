import pandas as pd
import numpy as np
import sys
import time
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

    def hypothesize(self, m):
        # initialize weights
        d = np.full(m, 1/m)

        # iterate over the number of weak classifiers we want
        for i in range(self.no_classifiers):
            train, test, attributes, labels = read_csv()
            train["weights"] = d

            # Decision Tree Stump
            h = DecisionTreeClassifier().create_tree(train, train, attributes, labels, gain="info_gain", maxdepth=2, use_weights=1)
            self.classifiers.append(h)

            # Calculate error
            train_pred = np.zeros(m, dtype=np.int32)
            for index, row in train.iterrows():
                pred = DecisionTreeClassifier().predict(row, h)
                if pred == "no":
                    train_pred[index] = -1
                else:
                    train_pred[index] = 1

            target = train["label"].copy().to_numpy()
            target[target == "no"] = -1
            target[target == "yes"] = 1

            #TODO: Calculate Error
            train_error = 0

            # Calculate alpha
            alpha = np.log((1-train_error)/train_error)/2

            # Update weights
            #TODO: Update weights

if __name__ == "__main__":
    ada = Adaboost()
    train, test, attributes, labels = read_csv()
    ada.hypothesize(train.shape[0])
    


