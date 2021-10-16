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

class RandomForest:
    def __init__(self, no_classifiers=5, G=2):
        self.no_classifiers = no_classifiers
        self.classifiers = []
        self.G = G

    def rf_fit(self):
        for i in range(self.no_classifiers):

            train, test, attributes, labels = read_csv()

            X = train.sample(train.shape[0], replace=True, ignore_index=True)
            attributes = np.random.choice(attributes, self.G, replace=False).tolist()
            
            tree = DecisionTreeClassifier().create_tree(X, X, attributes, labels, gain="info_gain", maxdepth=10000)

            self.classifiers.append(copy.copy(tree))
                
    def rf_predict(self, X):
        all_preds = []
        for cls in range(len(self.classifiers)):
            tree_pred = np.zeros(X.shape[0], dtype=np.float64)
            for index, row in X.iterrows():
                pred = predict(row, self.classifiers[cls])
                if pred == "no":
                    tree_pred[index] = -1
                else:
                    tree_pred[index] = 1
            all_preds.append(tree_pred)

        # Find mode of dataset
        final_prediction = pd.DataFrame(all_preds)
        final_prediction = final_prediction.T

        return final_prediction.mode(axis=1)[0]

if __name__ == "__main__":
    for size in range(1, 101):
        train, test, attributes, labels = read_csv()

        rf = RandomForest(no_classifiers=size, G=2)
        rf.rf_fit()
        test_pred = rf.rf_predict(test)
        train_pred = rf.rf_predict(train)

        # Calculate Testing Error
        target = test["label"].copy().to_numpy()
        target[target == "no"] = -1
        target[target == "yes"] = 1
        target = target.astype(float)

        errors = 0
        for i in range(len(target)):
            if target[i] != test_pred[i]:
                errors += 1

        test_error = errors / len(test)

        print(f"TEST ERROR {size}: {test_error}")

        # Calculate Train Error
        target = train["label"].copy().to_numpy()
        target[target == "no"] = -1
        target[target == "yes"] = 1
        target = target.astype(float)

        errors = 0
        for i in range(len(target)):
            if target[i] != train_pred[i]:
                errors += 1

        train_error = errors / len(train)

        print(f"TRAIN ERROR {size}: {train_error}")