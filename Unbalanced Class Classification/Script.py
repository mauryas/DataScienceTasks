# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 21:46:07 2018
These are the steps I have used on the analysis.
    - Exploring the data
    - Cleaning/pre-processing data
    - Training Models
    - Hypothesis Test

Please drop an email if you want further information 
at shivammaurya@outlook.com
	
@author: ShivamMaurya
"""
#%% Import Libraries
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from numba import jit

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from scipy.stats import friedmanchisquare
from scipy.stats import rankdata

from keras.models import Sequential
from keras.layers import (Dense, Dropout)
from keras.utils import to_categorical

#%% Class
class Analysis:
    def __init__(self,filepath):
        self.raw_data = self.read_data(filepath)
        self.X = []
        self.y = []
        
    def read_data(self,filepath):
        '''
        Read the file.
        '''        
        data = pd.read_csv(filepath,header=None)
        
        print('Data Loaded')
        return data
    @jit    
    def preprocess(self):
        '''
        Pre-processings:
            - remove unary features
            - normalize continous features
            - remove negative values for using Naive Bayes
            - Convert y into integers from alphabets
        '''
        
        X = self.raw_data
        y = self.raw_data[self.raw_data.shape[1]-1]
        
#        find indices of variables
        uni_idx = []#Unary
        bin_idx = []#Binary
        ter_idx = []#Ternary
        cont_idx = []#Continous
        for i in range(X.shape[1]-1):
            uniq_ = np.unique(X.values[:,i])
            if np.shape(uniq_)[0]==1:
                uni_idx.append(i)
            elif np.shape(uniq_)[0]==2:
                bin_idx.append(i)
            elif np.shape(uniq_)[0]>2 and np.shape(uniq_)[0]<1000:
                ter_idx.append(i)
            else:
                cont_idx.append(i)
        
#        remove unary features and class
        X = X.drop(labels=uni_idx,axis=1)
        X = X.drop(labels=295,axis=1)#remove the class 
        
        uniq_class = np.unique(y)
        
        j=0
        for i in uniq_class:
            y[y==i] = j
            j+=1
                
        #Sklearn libraries don't accept object type as y. So, convert it into int32 array
        y = np.array(y.values, dtype='int32')
        X = X.values
        
        X[X[:,23]<0,23] = 0#Naive Bayesian classifier doesnt take negetive values
        X[X[:,36]<0,36] = 0
        
        self.X = X
        self.y = y
        print('Data Pre-processed')
        
    def cross_validate(self,clf):
        '''
        Use cross-validation to train models and return their scores.
        input:
            clf : classifier
        output:
            scores : weighted F1 scores of 5 cv. 
        '''
        cv = StratifiedKFold(n_splits=5)
        scores = []
        
        for train, test in cv.split(self.X,self.y):
            X_train = self.X[train]
            y_train = self.y[train]
            
            X_test = self.X[test]
            y_test = self.y[test]
            #Train the classifier            
            clf.fit(X_train,y_train)
            
            #predict
            y_pred = clf.predict(X_test)
            
            #evaluate
            scores.append(f1_score(y_test,y_pred,average='weighted'))
        print(scores)
            
        return scores
    
    def rank_test(self,svm,nb,rf,nn):
        '''
        For ranking the classifiers, I will perform Friedman Rank Test. 
        The idea is to test are the average rank based on F1 score are 
        significantly different or not.
        H0 - F1 scores from classifiers are similar
        H1 - F1 scores from classifiers are different and ranking are significant.
        Input:
            svm - f1 scores from SVM
            nb -  f1 scores from Naive Bayes classifier
            rf -  f1 scores from Random Forests
            nn - f1 score from Neural Networks
        Output:
            avg. ranking: average rank based on f1 scores of the four classifiers
            pval: p-value from the Friedman Rank Test
        '''
        chi_sq, pval = friedmanchisquare(svm,nb,rf,nn)
        if pval<0.05:#alpha = 0.05
            print('Friedman Test: Reject Null Hypothesis with p-value: {}'.format(pval))
            ranking = np.zeros((len(svm),4))
            for i in range(len(svm)):
                result_slice = [svm[i],nb[i],rf[i],nn[i]]
                ranking[i] = rankdata(result_slice)
    
                return np.mean(ranking,axis=0), pval
        else:
            print('Friedman Test: Accept Null Hypothesis with p-value: {}'.format(pval))
            return [5,5,5,5], pval
    
    def neural_nets(self):
        '''
        This method will:
            1. Design a Deep Feed Forward Neural Network
            2. One-hot encode y
            3. train the NN model
            4. evaluate the model
        returns:
            scores: F1 scores from cross validation
        '''
        nclass = np.shape(np.unique(self.y))[0]
        input_dim = self.X.shape[1]
        
        #Design the computational graph
        model = Sequential()
        model.add(Dense(256,input_shape=(input_dim,),activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(256,activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(256,activation='sigmoid'))
        model.add(Dropout(0.5))
        model.add(Dense(nclass,activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        #training variables
        nbatch = 512
        nepochs = 100
        class_weights = {0:54.1, 1:7.1, #weights are calculated from data
                         2:1, 3:5.1, 4:18.7}
        
        cv = StratifiedKFold(n_splits=5)
        scores = []
        
        #One-Hot encode y
        for train, test in cv.split(self.X,self.y):
            X_train = self.X[train]
            y_train = self.y[train]
            
            X_test = self.X[test]
            y_test = self.y[test]
            
            y_train_onehot = to_categorical(y_train)
            y_test_onehot = to_categorical(y_test)
                
            history = model.fit(X_train, y_train_onehot, 
                                batch_size=nbatch, epochs=nepochs, 
                                verbose=0,class_weight=class_weights, 
                                validation_data=(X_test, y_test_onehot))
            
            y_pred = model.predict(X_test)
            
            pred_class=[]
            for i in y_pred:
                pred_class.append(np.argmax(i))
                
            #evaluate
            scores.append(f1_score(y_test,pred_class,average='weighted'))
        
        print(scores)
        return model, scores
       
#%% Analysis
def main():
    #Path of the csv file
    filepath = ''
    #Start Analysis
    analysis = Analysis(filepath)
    
    #Preprocess data
    analysis.preprocess()
    
    #Train a linear svm
    print('Linear SVM')
    clf_svm = LinearSVC(class_weight='balanced')
    score_svm = analysis.cross_validate(clf_svm)
    
    #Train Naive Bayes classifier    
    print('Naive Bayes')
    clf_nb = MultinomialNB()
    score_nb = analysis.cross_validate(clf_nb)
    
    #Train Random Forest
    print('Random Forest')
    clf_rf = RandomForestClassifier(n_estimators = 20,
                                class_weight='balanced',
                                n_jobs =-1)
    score_rf = analysis.cross_validate(clf_rf)
    
    #Design and train NN
    print('Neural Nets')
    clf_nn, score_nn = analysis.neural_nets()
    
    ranking, pval = analysis.rank_test(score_svm,score_nb,
                                             score_rf,score_nn)
    
    models=['Linear SVM', 'Naive Bayesian', 'Random Forests', 'Neural Network']
    
    print('Best Model is: {}'.format(models[np.argmax(ranking)]))
        
    
if __name__ == "__main__":
    main()    