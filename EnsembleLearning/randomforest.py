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

def predict(row, root):
    if root.value == "no":
        return "no"
    
    if root.value == "yes":
        return "yes"
        
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

    def rf_fit(self, train, attributes, labels):
        for i in range(self.no_classifiers):
            X = train.sample(train.shape[0], replace=True, ignore_index=True)
            sample_atr = np.random.choice(attributes, self.G, replace=False).tolist()
            
            tree = DecisionTreeClassifier().create_tree(X, X, sample_atr, labels, gain="info_gain", maxdepth=10000)

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

def singletree_bv(predictors, target):
    # using single trees
    single_trees = []
    for p in predictors:
        single_trees.append(copy.copy(p.classifiers[0]))

    single_predictions = []
    for t in single_trees:
        temp_pred = np.zeros(test.shape[0])
        for index, row in test.iterrows():
            pred = predict(row, t)
            if pred == "no":
                temp_pred[index] = -1
            else:
                temp_pred[index] = 1
        single_predictions.append(temp_pred)

    # Calculating Bias
    avg_single_predictions = np.mean(single_predictions, axis=0)

    bias = np.square(avg_single_predictions - target)

    # Calculating Variance
    variance = np.var(single_predictions, axis=0)

    # General bias and variance
    general_bias = np.mean(bias)
    general_variance = np.mean(variance)
    general_sqerror = general_bias + general_variance

    print(general_bias)
    print(general_variance)
    print(general_sqerror)

def randomforest_bv(predictors, target):
    rf_predictions = []
    for p in predictors:
        train, test, attributes, labels = read_csv()
        temp_pred = p.rf_predict(test)
        rf_predictions.append(temp_pred)

    # Calculating Bias
    avg_rf_predictions = np.mean(rf_predictions, axis=0)

    bias = np.square(avg_rf_predictions - target)

    # Calculating Variance
    variance = np.var(rf_predictions, axis=0)

    # General bias and variance
    general_bias = np.mean(bias)
    general_variance = np.mean(variance)
    general_sqerror = general_bias + general_variance

    print(general_bias)
    print(general_variance)
    print(general_sqerror)

if __name__ == "__main__":
    # data_upload_test = []
    # data_upload_train = []
    # for size in range(1, 101):
    #     train, test, attributes, labels = read_csv()

    #     rf = RandomForest(no_classifiers=size, G=6)
    #     rf.rf_fit(train, attributes, labels)
    #     test_pred = rf.rf_predict(test)
    #     train_pred = rf.rf_predict(train)

    #     # Calculate Testing Error
    #     target = test["label"].copy().to_numpy()
    #     target[target == "no"] = -1
    #     target[target == "yes"] = 1
    #     target = target.astype(float)

    #     errors = 0
    #     for i in range(len(target)):
    #         if target[i] != test_pred[i]:
    #             errors += 1

    #     test_error = (errors / len(test))*100

    #     print(f"TEST ERROR {size}: {test_error}")
    #     data_upload_test.append(test_error)

    #     # Calculate Train Error
    #     target = train["label"].copy().to_numpy()
    #     target[target == "no"] = -1
    #     target[target == "yes"] = 1
    #     target = target.astype(float)

    #     errors = 0
    #     for i in range(len(target)):
    #         if target[i] != train_pred[i]:
    #             errors += 1

    #     train_error = (errors / len(train))*100

    #     print(f"TRAIN ERROR {size}: {train_error}")
    #     data_upload_train.append(train_error)

    # UNCOMMENT IF WRITE TO FILE
    # with open('rf_test.txt', 'w') as f:
    #     for item in data_upload_test:
    #         f.write("%s\n" % item)

    # with open('rf_train.txt', 'w') as f:
    #     for item in data_upload_train:
    #         f.write("%s\n" % item)

    #Uncomment this section to for bias/variance tradeoff experiment between bagged trees and single trees
    #Bias and Variance decomposition experiment
    predictors = []
    for i in range(1, 3):
        train, test, attributes, labels = read_csv()
        sample = train.sample(1000, replace=False, ignore_index=True)
        rf = RandomForest(no_classifiers=2)
        rf.rf_fit(train, attributes, labels)
        predictors.append(copy.copy(rf))

    train, test, attributes, labels = read_csv()

    target = test["label"].copy().to_numpy()
    target[target == "no"] = -1
    target[target == "yes"] = 1
    target = target.astype(float)

    singletree_bv(predictors, target)
    randomforest_bv(predictors, target)