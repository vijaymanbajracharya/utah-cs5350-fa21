# ID3 algorithm with information gain
import pandas as pd
import numpy as np

class Node:
    def __init__(self, value, next=None):
        self.value = value
        self.next = next

class DecisionTree:

    def entropy(self, attributes):
        pass

    def information_gain(self, train):
        pass

    def id3(self, train, attributes, labels):
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



