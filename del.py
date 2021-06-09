from data_preprocess import data_preprocess
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.model_selection import train_test_split
from disease_prediction.decision_tree_scratch import DecisionTree
import numpy as np
import pickle
import argparse
from data_preprocess_util import dataArrange
from collections import Counter
# import dill

def train():
    x = data_preprocess()
    x.to_csv('Dataset.csv')

    y_actual = x.index
    x.reset_index(inplace= True, drop = True)
    encode = LabelEncoder()

    y = encode.fit_transform(y_actual)
    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy
    clf_array = []
    for count in range(10):
        X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = .2, train_size = .8)
        X_train_val = X_train.values
        X_test_val =X_test.values
        clf = DecisionTree()
        clf.fit(X_train_val,y_train)
        clf_array.append(clf)
    
    

    y_pred = clf.predict(X_train_val)
    acc = accuracy(y_train, y_pred)
    print ("Accuracy of train values:", acc)

    y_pred = clf.predict(X_test_val)
    acc = accuracy(y_test, y_pred)
    print ("Accuracy of test values:", acc)


    with open('model','wb') as model_file:
        pickle.dump(clf_array,model_file)

    # with open('dill_model','wb') as model_file:
    #     dill.dump(clf_array,model_file)

    with open('encode_model','wb') as model_file:
        pickle.dump(encode,model_file)

def prediction(X_raw):
    sympfile = open('symp_model','rb')
    symp_list = pickle.load(sympfile)

    infile = open('model','rb')
    clf = pickle.load(infile)

    encode_file = open('encode_model','rb')
    encode = pickle.load(encode_file)

    X = dataArrange(symp_list , X_raw)
    X = [X]

    pred_array = []
    for count in range(10):
        pred = (clf[count].predict(X))
        pred = encode.inverse_transform(pred)
        pred_array.append(pred[0])
    

    counter = Counter(pred_array)
    iterator = counter.most_common()

    disease_prediction = []

    for i in iterator:
      
        probability = i[1]
        probability/=10
        probability *= 100        
        temp = { "disease" : i[0], "probability" : probability}
        disease_prediction.append(temp)
    
    print(disease_prediction)

    infile.close
    sympfile.close
    encode_file.close

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('function' , help='(train) - to train a model (predict) - to predict the disease')
    parser.add_argument('--symptoms' ,nargs='+', help='List of symptopms', default="")

    args = parser.parse_args()
    if (args.function == 'train'):
        train()
    elif (args.function == 'predict'):
        if (args.symptoms == ""):
            print("Please input symptoms")
        else:
            prediction(args.symptoms)
    else :
        print("Invalid Inputs, try typing (del.py -h) for help") 