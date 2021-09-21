# ID3 algorithm with information gain
import pandas as pd
import numpy as np

MAX_DEPTH = 10

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
        atr_entropy = self.cross_entropy(subset.iloc[:, np.r_[index,6]], atr_name)

        # DEBUG PORTION
        self.ME_gain(subset, atr_name)

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
        atr_ME = self.majority_error(subset.iloc[:, np.r_[index,6]], atr_name)
        
        return total_ME - atr_ME

    def gini_index():
        pass

    def gini_gain(self, subset, atr_name):
        # Calculate the gini index of the whole subset
        pass


    def create_tree(self, data, train, attributes, labels, node=None, depth=0):
        # Base case check for same labels
        global MAX_DEPTH
        if node is None:
            node = Node()

        values, counts = np.unique(data["label"], return_counts=True)
        if len(counts) <= 1:
            # Return the label as a 
            node.value = values[0]
            return node
        # Base case check for empty attributes
        elif len(attributes) < 1:
            # Return most common label
            node.value = values[np.argmax(counts)]
            return node
        elif depth == MAX_DEPTH:
            node.value = values[np.argmax(counts)]
            return node
        else:
            # Find attribute that gives maximum information gain
            info_gain = {}
            for atr_name in attributes:
                info_gain[atr_name] = self.information_gain(data, atr_name)
            best_attribute = max(info_gain, key=info_gain.get)

            # Create tree with best_attribute at root
            node.value = best_attribute
            node.edge = []

            # Remove best_attribute from attributes list
            attributes.remove(best_attribute)

            for branch in np.unique(data[best_attribute]):
                child = Node()
                child.feature_value = branch
                node.edge.append(child)
                subset = data.loc[data[best_attribute] == branch]
                child = self.create_tree(subset, train, attributes, labels, child, depth=depth+1)

            return node

    def predict(self, test, root):
        pass


if __name__ == "__main__":
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

    train = pd.read_csv("utah-cs5350-fa21/DecisionTree/train.csv", names=table)
    test = pd.read_csv("utah-cs5350-fa21/DecisionTree/test.csv", names=table)

    root = DecisionTreeClassifier().create_tree(train, train, attributes, labels)
    pass



