#!/usr/bin/env python3

'''
Skeleton for Homework 4: Logistic Regression and Decision Trees
Part 2: Decision Trees

Authors: Anja Gumpinger, Bastian Rieck
'''

import numpy as np
import sklearn.datasets
from sklearn.datasets import load_iris

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


##2A

def split_data(X, y, attribute_index, theta):
    
    attribute=X[:,attribute_index] #select the indecx to be the reference for the split
    d1=np.where(attribute<theta) #see which values are smaller than theta
    d2=np.where(attribute>=theta) #see which values are bigger or equal than theta
    
    d1_y=y[d1]
    d2_y=y[d2]
    
    return d1_y, d2_y

def compute_information_content(y):
    
    y_unique=np.unique(y) #choose unique target names
    
    infromation_content=0
    
    for y_i in y_unique:
        p_i=np.sum(y==y_i)/y.shape[0] #calculate the percentage of occurences for each target
        infromation_content+=p_i*np.log2(p_i) #apply the formula
        
    infromation_content=-1*infromation_content
        
    return infromation_content

def compute_information_a(X, y, attribute_index, theta):
    
    d1_y, d2_y=split_data(X, y, attribute_index, theta) #split the data with according to attribute and theta selected
    d_ys=[d1_y, d2_y] #save results as a list
    
    infromation_content_a_d=0
    
    for d_yj in d_ys: #iterate over the list to calculate infromation_content_a_d
        
        infromation_content_d_j=compute_information_content(d_yj)
        term=d_yj.shape[0]*infromation_content_d_j/y.shape[0]
        
        infromation_content_a_d+=term
        
    return infromation_content_a_d
        
        
def compute_information_gain(X, y, attribute_index, theta):
    
    info_d=compute_information_content(y)
    info_a_d=compute_information_a(X, y, attribute_index, theta)
    
    gain_a=info_d-info_a_d #after computing information contents substract them to geth the gains
    
    return(gain_a)
 
##2D

#split the data for cross validation

def find_mean_accuracy_and_feature_importance(X, y, number_of_features):
    
    kf=KFold(n_splits=5, shuffle=True)
    
    accuracy=np.empty(0)
    feature_importance=[]
    
    for train_index, test_index in kf.split(X, y):
        
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        
        
        dtc.fit(X_train,y_train)
        y_pred=dtc.predict(X_test)
        
        feature_importance.append(np.argsort(dtc.feature_importances_))
        
        fold_accuracy=accuracy_score(y_test, y_pred)
        accuracy=np.append(accuracy, fold_accuracy)
        
    mean_accuracy=round(np.mean(accuracy)*100,2)
    
    most_important_features=[]
    
    for importance in feature_importance:
        most_important_features.append(importance[-number_of_features:])
    
    return mean_accuracy, most_important_features

if __name__ == '__main__':
    
    # to load the data into X and labels into y
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # to see feature names and label names:
    feature_names = iris.feature_names
    target_names = iris.target_names


    ####################################################################
    # Your code goes here.
    ####################################################################
    conditions=[[0,5.5],[1,3], [2,2],[3,1]]
    
    gains=[]
    
    for c in conditions:
        
        attribute_index=c[0]
        theta=c[1]
        
        output=round(compute_information_gain(X, y, attribute_index, theta),2)
        gains.append(output)
        
    gains=np.array(gains)
    
    print('Exercise 2.b')
    print('------------')
    print('Split ( sepal length (cm) < 5.5 ): information gain = {}'.format(gains[0]))
    print('Split ( sepal width (cm) < 3.0 ): information gain = {}'.format(gains[1]))
    print('Split ( petal length (cm) < 2.0 ): information gain = {}'.format(gains[2]))
    print('Split ( petal width (cm) < 1.0 ): information gain = {}'.format(gains[3]))
    print('')
    
    #The one with maximum information gain therefore:
    best_split_value=np.array(conditions,dtype=int)[np.where(np.max(gains) == gains)]


    print('Exercise 2.c')
    print('------------')
    print('I would select {one} < {two} or {three} < {four} to be the first split, because they both provide the highest information gain'.format(one=feature_names[best_split_value[0][0]],two=float(best_split_value[0][1]),three=feature_names[best_split_value[1][0]],four=float(best_split_value[1][1])))
    print('')
    
    np.random.seed(42)
    dtc=DecisionTreeClassifier()
    
    full_data=find_mean_accuracy_and_feature_importance(X,y,2)

    X_without_2= X[y != 2]
    y_without_2= y[y != 2]

    reduced_data=find_mean_accuracy_and_feature_importance(X_without_2,y_without_2,1)

    ####################################################################
    print("Exercise 2.d")
    ####################################################################



    print('The mean accuracy is {}'.format(full_data[0]))


    print('')
    print('For the original data, the two most important features are:')
    print('-{}'.format(feature_names[np.unique(full_data[1])[0]]))
    print('-{}'.format(feature_names[np.unique(full_data[1])[1]]))


    print('')
    print('For the reduced data, the most important feature is:')
    print('-{}'.format(feature_names[np.unique(reduced_data[1])[0]]))
    print('This means that {} is really similar within the eliminated species and is significantly different among the three species. Therefore the eliminated species provides lots of insight about this attribute and influences the structure of the tree if it is considred and not eliminated'.format(feature_names[np.unique(full_data[1])[1]]))

     