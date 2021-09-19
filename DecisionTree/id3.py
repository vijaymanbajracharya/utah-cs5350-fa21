# ID3 algorithm with information gain
import pandas as pd
import numpy as np

class Node:
    def __init__(self):
        self.value = None
        self.edge = None

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

        # Return the information gain
        return total_entropy - atr_entropy


    def create_tree(self, data, train, attributes, labels, node=None, depth=None):
        # Base case check for same labels
        values, counts = np.unique(data["label"], return_counts=True)
        if len(counts) <= 1:
            # Return the label as a 
            if not node:
                node = Node()
            node.value = values[0]
            return node
        # Base case check for empty attributes
        elif len(attributes) < 1:
            # Return most common label
            if not node:
                node = Node()
            node.value = values[np.argmax(counts)]
            return node
        else:
            if not node:
                node = Node()

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
                node.edge.append(child)
                subset = data.loc[data[best_attribute] == branch]
                child = self.create_tree(subset, train, attributes, labels, child, depth)

            return node


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

    tree = DecisionTreeClassifier().create_tree(train, train, attributes, labels)



