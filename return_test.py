import numpy as np
from decision_tree_test import return_test 

clf = return_test()
X = [[1,0,0]]
print(clf.predict(X))