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

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy



diseases_list=['caninedistemper', 'canineparvovirus', 'heartworm', 'kennelcough','kidneydisease','leptospirosis','lymedisease','rabies']
        
        
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
    

    
    
    X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = .2, train_size = .8)
    X_train_val = X_train.values
    X_test_val =X_test.values
    
     
    model = Sequential([
        Dense(units=16, input_shape=(57,), activation='relu'),
        Dense(units=32, activation='relu'),
#       Dense(units=16,activation='relu'),
        Dense(units=8, activation='softmax')
        ])
    
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    model.fit(x=X_train_val, y=y_train, batch_size=10, epochs=250, verbose=2)
    
    y_pred = model.predict(X_train_val)
    acc = accuracy(y_train, np.argmax(y_pred,axis=-1))
    print ("Accuracy of train values:", acc)

    y_pred = model.predict(X_test_val)
    acc = accuracy(y_test,np.argmax(y_pred,axis=-1))
    print ("Accuracy of test values:", acc)

    def test(test_x,test_y):   # only for testing
        max_probs=[]
        preds=[]
    
        y_pred = model.predict(test_x)
        acc = accuracy(test_y,np.argmax(y_pred,axis=-1))
        print ("Accuracy of test values:", acc)
    
    
        for i in range(len(y_pred)):
            prob=max(y_pred[i])
            max_probs.append(prob)
        
            pred = np.argmax(y_pred[i], axis=-1)
            preds.append(pred)
    
            print(f"{i+1} Disease: {diseases_list[pred]}, {pred} , Probability: {prob}")


    # test(X_test_val,y_test)

    model.save('tf_model')
    
    # with open('model','wb') as model_file:
    #     pickle.dump(model,model_file)

#    # with open('dill_model','wb') as model_file:
#     #     dill.dump(model,model_file)

    # with open('encode_model','wb') as model_file:
    #     pickle.dump(encode,model_file)

def prediction(X_raw):
    sympfile = open('symp_model','rb')
    symp_list = pickle.load(sympfile)

    # infile = open('model','rb')
    # clf = pickle.load(infile)
    model=tf.keras.models.load_model('tf_model')

    # encode_file = open('encode_model','rb')
    # encode = pickle.load(encode_file)

    X = dataArrange(symp_list , X_raw)
    X = [X]

    
    pred = model.predict(X)
    pred_acc=np.argmax(pred,axis=-1)
    prob=max(pred)


    disease_prediction = []

    temp = { "disease" : diseases_list[pred_acc], "probability" : prob}
    disease_prediction.append(temp)
    
    print(disease_prediction)

    sympfile.close

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
