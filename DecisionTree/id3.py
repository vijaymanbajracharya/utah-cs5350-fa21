# ID3 algorithm with information gain
import pandas as pd
import numpy as np

class Node:
    def __init__(self, value, next=None):
        self.value = value
        self.next = next

class DecisionTree:

    def cross_entropy(self, attribute, atr_name):
        freq = attribute.groupby([atr_name,"label"]).size().reset_index(name="Count")
        pass

    def information_gain(self, subset, atr_name):
        index = subset.columns.get_loc(atr_name)
        self.cross_entropy(subset.iloc[:, np.r_[index,6]], atr_name)
        pass

    def id3(self, subset, train, attributes, labels):
        info_gain = []
        for atr_name in attributes:
            info_gain.append(self.information_gain(subset, atr_name))

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



