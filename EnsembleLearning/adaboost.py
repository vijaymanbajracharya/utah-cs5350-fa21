from os import error
import pandas as pd
import numpy as np
import sys
import time
import copy
sys.path.append("../")
#sys.path.append("utah-cs5350-fa21/")
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

    train = pd.read_csv("bank/train.csv", names=table)
    test = pd.read_csv("bank/test.csv", names=table)
    #train = pd.read_csv("utah-cs5350-fa21/DecisionTree/bank/train.csv", names=table)
    #test = pd.read_csv("utah-cs5350-fa21/DecisionTree/bank/test.csv", names=table)

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

def predict(row, root):
        value = row[root.value]
        for branches in root.edge:
            if branches.feature_value == value:
                if branches.edge is None:
                    pred = branches.value
                else:
                    pred = predict(row, branches)
        return pred

class Adaboost:
    def __init__(self, no_classifiers=5):
        self.no_classifiers = no_classifiers
        self.classifiers = []
        self.alphas = []
        self.preds = []

    def ada_fit(self, m):
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
                pred = predict(row, h)
                if pred == "no":
                    train_pred[index] = -1
                else:
                    train_pred[index] = 1

            target = train["label"].copy().to_numpy()
            target[target == "no"] = -1
            target[target == "yes"] = 1
            target = target.astype(float)

            weights = train["weights"].copy().to_numpy()
            weights = weights.astype(float)

            errors = 0.0
            for i in range(len(target)):
                if target[i] != train_pred[i]:
                    errors += weights[i]

            self.preds.append(train_pred)

            # Calculate alpha
            alpha = 0.5 * np.log((1-errors)/errors)
            self.alphas.append(copy.copy(alpha))

            # Update weights
            d = d * np.exp(-alpha * target * train_pred)
            d = d / np.sum(d)

    def ada_predict(self, X):
        test_pred = []
        for cls in range(self.no_classifiers):
            prediction = np.zeros(test.shape[0], dtype=np.float64)
            for index, row in test.iterrows():
                pred = predict(row, self.classifiers[cls])
                if pred == "no":
                    prediction[index] = -1
                else:
                    prediction[index] = 1

            test_pred.append(prediction)
    
        y_pred = np.sign(np.dot(self.alphas, test_pred))

    

if __name__ == "__main__":
    data_upload_test = []
    data_upload_train = []
    for size in range(1, 201):
        train, test, attributes, labels = read_csv()

        # Train 
        ada = Adaboost(no_classifiers=size)
        ada.ada_fit(train.shape[0])
        train_pred = np.sign(np.dot(ada.alphas, ada.preds))
        test_pred = ada.ada_predict(test)

        # Calculate Testing Error
        target = test["label"].copy().to_numpy()
        target[target == "no"] = -1
        target[target == "yes"] = 1
        target = target.astype(float)

        errors = 0
        for i in range(len(target)):
            if target[i] != train_pred[i]:
                errors += 1

        test_error = (errors / len(test))*100
        print(f"TEST ERROR {size}: {test_error}")
        data_upload_test.append(test_error)


        # Calculate Trainning Error
        target = train["label"].copy().to_numpy()
        target[target == "no"] = -1
        target[target == "yes"] = 1
        target = target.astype(float)

        errors = 0
        for i in range(len(target)):
            if target[i] != train_pred[i]:
                errors += 1

        train_error = (errors / len(train))*100
        print(f"TRAIN ERROR {size}: {train_error}")
        data_upload_train.append(train_error)
    
    with open('ada_test.txt', 'w') as f:
        for item in data_upload_test:
            f.write("%s\n" % item)

    with open('ada_train.txt', 'w') as f:
        for item in data_upload_train:
            f.write("%s\n" % item)

    



