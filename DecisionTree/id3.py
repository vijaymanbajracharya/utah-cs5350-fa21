# ID3 algorithm with information gain
import pandas as pd
import numpy as np

class Node:
    def __init__(self, value, next=None):
        self.value = value
        self.next = next

class DecisionTree:

    def cross_entropy(self, attribute, atr_name):
        num_train = attribute.shape[0]

        freq_by_atr = attribute.groupby([atr_name]).size().reset_index(name="Count")
        freq_by_row = attribute.groupby([atr_name,"label"]).size().reset_index(name="Count")

        total_entropy = 0.0
        feature_counts = {}

        for index, row in freq_by_atr.iterrows():
            feature_counts[row[atr_name]] = row["Count"] 

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


    def id3(self, subset, train, attributes, labels):
        # Base case check for same labels
        values, counts = np.unique(subset["label"], return_counts=True)
        if len(counts) <= 1:
            return values[0]
        
        # Find attribute that gives maximum information gain
        info_gain = {}
        for atr_name in attributes:
            info_gain[atr_name] = self.information_gain(subset, atr_name)

        print(info_gain)

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

    DecisionTree().id3(train, train, attributes, labels)



