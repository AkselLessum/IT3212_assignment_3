#Decision trees
#Random Forest
#Neural Network (er lættis)
#SVM with kernels (viable)
#Additive models

#decision trees
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#decision trees 
def decisionTrees():
    #load data
    data = pd.read_csv('../graduation_dataset.csv')
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #create model
    model = tree.DecisionTreeClassifier()
    model.fit(X_train, y_train)
    #predict
    y_pred = model.predict(X_test)
    print("Decision Trees accuracy: ", accuracy_score(y_test, y_pred))
    
#random forest
def randomForest():
    #load data
    data = pd.read_csv('../graduation_dataset.csv')
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #create model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    #predict
    y_pred = model.predict(X_test)
    print("Random Forest accuracy: ", accuracy_score(y_test, y_pred))

#Additive model
def additiveModel():
    #load data
    data = pd.read_csv('../graduation_dataset.csv')
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #create model
    model = MLPClassifier()
    model.fit(X_train, y_train)
    #predict
    y_pred = model.predict(X_test)
    print("Additive model accuracy: ", accuracy_score(y_test, y_pred))  

#SVM with kernels
def svmKernels():
    #load data
    data = pd.read_csv('../graduation_dataset.csv')
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #create model
    model = SVC(kernel='rbf')
    model.fit(X_train, y_train)
    #predict
    y_pred = model.predict(X_test)
    print("SVM with kernels accuracy: ", accuracy_score(y_test, y_pred))
    
#Neural Network
def neuralNetwork():
    #todo
    pass

 