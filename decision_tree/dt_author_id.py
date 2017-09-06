#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from time import time
import numpy as np


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
print(features_train.shape)

from sklearn import tree
clf = tree.DecisionTreeClassifier(min_samples_split=40)
t = time()
clf.fit(features_train, labels_train)
print('train time=%.2f' % (time() - t))

t = time()
pred = clf.predict(features_test)
print('predict time=%.2f' % (time() - t))

acc = 1 - np.abs(pred-labels_test).mean()
print acc
