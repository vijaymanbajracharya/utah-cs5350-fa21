# ID3 algorithm with information gain
import pandas as pd
import numpy as np
import time

MAX_DEPTH = 16

class Node:
    def __init__(self):
        self.value = None
        self.edge = None
        self.feature_value = None

class DecisionTreeClassifier:
    def cross_entropy(self, attribute, atr_name):
        # Get the number of rows
        num_train = attribute.shape[0]

        # Grouping to make it easier to calculate probabilities
        freq_by_atr = attribute.groupby([atr_name]).size().reset_index(name="Count")
        freq_by_row = attribute.groupby([atr_name,"label"]).size().reset_index(name="Count")

        total_entropy = 0.0
        feature_counts = {}

        # Count the total for each value
        for index, row in freq_by_atr.iterrows():
            feature_counts[row[atr_name]] = row["Count"] 

        # Calculate entropy of attribute based on value
        for index, row in freq_by_row.iterrows():
            prob = row["Count"]/feature_counts[row[atr_name]]
            entropy = -prob*np.log2(prob)
            total_entropy += entropy * (feature_counts[row[atr_name]]/num_train)

        return total_entropy

    def information_gain(self, subset, atr_name):
        # Calculate entropy of the whole subset
        values, counts = np.unique(subset["label"], return_counts=True)
        sum_counts = np.sum(counts)
        total_entropy = 0.0
        for i in range(len(counts)):
            prob = counts[i]/sum_counts
            total_entropy += -prob*np.log2(prob)

        # Calculate entropy of particular attribute
        index = subset.columns.get_loc(atr_name)
        label_index = subset.columns.get_loc("label")
        atr_entropy = self.cross_entropy(subset.iloc[:, np.r_[index,label_index]], atr_name)

        # Return the information gain
        return total_entropy - atr_entropy

    def majority_error(self, attribute, atr_name):
        # Get the number of rows
        num_train = attribute.shape[0]

        # Grouping to make it easier to calculate probabilities
        freq_by_atr = attribute.groupby([atr_name]).size().reset_index(name="Count")
        freq_by_row = attribute.groupby([atr_name,"label"]).size().reset_index(name="Count")

        total_ME = 0.0
        feature_counts = {}

        # Count the total for each value
        for index, row in freq_by_atr.iterrows():
            feature_counts[row[atr_name]] = row["Count"] 

        # Find the max of each feature value
        for keys in feature_counts.keys():
            freq_by_atrvalue = freq_by_row.loc[freq_by_row[atr_name] == keys]
            majority = freq_by_atrvalue["Count"].max()
            error = (feature_counts[keys]-majority)/feature_counts[keys]
            total_ME += error * (feature_counts[keys]/num_train) 

        return total_ME

    def ME_gain(self, subset, atr_name):
        # Calculate the majority error of the whole subset
        values, counts = np.unique(subset["label"], return_counts=True)
        majority = max(counts)
        sum_counts = np.sum(counts)
        total_ME = (sum_counts-majority) / sum_counts

        # Calculate ME of particular attribute
        index = subset.columns.get_loc(atr_name)
        label_index = subset.columns.get_loc("label")
        atr_ME = self.majority_error(subset.iloc[:, np.r_[index,label_index]], atr_name)
        
        return total_ME - atr_ME

    def gini_index(self, attribute, atr_name):
         # Get the number of rows
        num_train = attribute.shape[0]

        # Grouping to make it easier to calculate probabilities
        freq_by_atr = attribute.groupby([atr_name]).size().reset_index(name="Count")
        freq_by_row = attribute.groupby([atr_name,"label"]).size().reset_index(name="Count")

        total_gini = 0.0
        feature_counts = {}

        # Count the total for each value
        for index, row in freq_by_atr.iterrows():
            feature_counts[row[atr_name]] = row["Count"]
        
        for keys in feature_counts.keys():
            freq_by_atrvalue = freq_by_row.loc[freq_by_row[atr_name] == keys]
            sqaured_sum = 0.0
            for index, row in freq_by_atrvalue.iterrows():
                prob = row["Count"]/feature_counts[row[atr_name]]
                sqaured_sum += np.square(prob)
            
            total_gini += (1-sqaured_sum) * (feature_counts[row[atr_name]]/num_train)

        return total_gini

        

    def gini_gain(self, subset, atr_name):
        # Calculate the gini index of the whole subset
        values, counts = np.unique(subset["label"], return_counts=True)
        sum_counts = np.sum(counts)
        squared_sum = 0.0
        for i in range(len(counts)):
            prob = counts[i]/sum_counts
            squared_sum += np.square(prob)
        
        total_gini = 1 - squared_sum

        # Calculate gini index of particular attribute
        index = subset.columns.get_loc(atr_name)
        label_index = subset.columns.get_loc("label")
        atr_gini = self.gini_index(subset.iloc[:, np.r_[index,label_index]], atr_name)

        return total_gini - atr_gini


    def create_tree(self, data, train, attributes, labels, node=Node(), depth=0, gain="info_gain"):
        # Base case check for same labels
        global MAX_DEPTH
        values, counts = np.unique(data["label"], return_counts=True)
        if len(counts) <= 1:
            # Return the label as a node 
            node.value = values[0]
            return node
        # Base case check for empty attributes
        elif len(attributes) < 1:
            # Return most common label
            node.value = values[np.argmax(counts)]
            return node
        # Check if we are at max depth
        elif depth == MAX_DEPTH:
            node.value = values[np.argmax(counts)]
            return node
        else:
            # Find attribute that gives maximum information gain
            info_gain = {}
            for atr_name in attributes:
                if gain == "majority_error":
                    info_gain[atr_name] = self.ME_gain(data, atr_name)
                elif gain == "gini_index":
                    info_gain[atr_name] = self.gini_gain(data, atr_name)
                else:
                    info_gain[atr_name] = self.information_gain(data, atr_name)
            best_attribute = max(info_gain, key=info_gain.get)

            # Create tree with best_attribute at root
            node.value = best_attribute
            node.edge = []

            if best_attribute in attributes:
                # Remove best_attribute from attributes list
                attributes.remove(best_attribute)

            for branch in np.unique(train[best_attribute]):
                # Create a leaf child
                child = Node()
                child.feature_value = branch
                node.edge.append(child)

                # Split the dataset on the best attribute
                subset = data.loc[data[best_attribute] == branch]

                # If the split contains no rows
                if len(subset) == 0:
                    child.value = values[np.argmax(counts)]
                else:
                    child = self.create_tree(subset, train, attributes.copy(), labels, child, depth=depth+1, gain=gain)

            return node

    def predict(self, row, root):
        value = row[root.value]
        for branches in root.edge:
            if branches.feature_value == value:
                if branches.edge is None:
                    pred = branches.value
                else:
                    pred = self.predict(row, branches)
        return pred
    
    def error(self, pred, data):
        num_correct = 0
        total = data.shape[0]
        for index, row in data.iterrows():
            if pred[index] == row['label']:
                num_correct += 1
        return 1-num_correct/total

def find_error(test, train, root):
    test_pred = {}
    for index, row in test.iterrows():
        test_pred[index] = DecisionTreeClassifier().predict(row, root)

    test_error = DecisionTreeClassifier().error(test_pred, test)
    print(f"Test Error: {test_error*100}")

    train_pred = {}
    for index, row in train.iterrows():
        train_pred[index] = DecisionTreeClassifier().predict(row, root)

    train_error = DecisionTreeClassifier().error(train_pred, train)
    print(f"Train Error: {train_error*100}")

def decision_tree_car(gain="info_gain"):
    cols = '''buying,
        maint,
        doors,
        persons,
        lug_boot,
        safety,
        label'''
    table=[]
    labels = ["unacc", "acc", "good", "vgood"]
    attributes = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]
    for c in cols.split(','):
        if(c.strip()):
            table.append(c.strip())

    train = pd.read_csv("train.csv", names=table)
    test = pd.read_csv("test.csv", names=table)

    root = DecisionTreeClassifier().create_tree(train, train, attributes, labels, gain=gain)

    print(f"Prediction errors % - Car ({gain})")
    find_error(test, train, root)
    print("\r\n")

def decision_tree_bank(gain="info_gain"):
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

    # Turning missing attribute "unknown" into most common value
    for atr in attributes:
        train_col = train.loc[:,atr]
        if any(train_col == "unknown"):
            values, counts = np.unique(train_col, return_counts=True)
            mcv_index = np.argmax(counts)
            if values[mcv_index] == "unknown":
                mcv_index = np.argsort(counts)[-2]

            mcv = values[mcv_index]
            test_col = test.loc[:,atr]

            train.loc[train[atr] == "unknown", atr] = mcv
            test.loc[test[atr] == "unknown", atr] = mcv
        else:
            continue


    root = DecisionTreeClassifier().create_tree(train, train, attributes, labels, gain=gain)

    print(f"Prediction errors % - Bank ({gain})")
    find_error(test, train, root)
    print("\r\n")

if __name__ == "__main__":
    pd.options.mode.chained_assignment = None

    print("\r\nThe default max-depth is 16. Modify MAX_DEPTH global to change height of decision tree.\r\n")

    heuristics = ["info_gain", "majority_error", "gini_index"]
    for h in heuristics:
        decision_tree_car(h)

    for h in heuristics:
        decision_tree_bank(h)


    

        
