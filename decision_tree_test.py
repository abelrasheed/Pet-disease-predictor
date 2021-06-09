import numpy as np
from decision_tree_scratch import DecisionTree


def return_test():
     x = [[0,0,1],
          [0,1,0],
          [0,1,1],
          [1,0,0],
          [0,0,0],
          [1,1,0],
          [1,1,1]]
     X = np.array(x)
     y = [0,0,1,1,0,1,1]
     Y = np.array(y)

     clf = DecisionTree()
     clf.fit(X,Y)
     return clf