#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn import svm
import numpy as np

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
#L = 100
#features_train, labels_train = features_train[:L,:], labels_train[:L]
#n_samples = features_train.shape[0]
#features_train, labels_train = features_train[:n_samples/100,:], labels_train[:n_samples/100]

'''
C_vec = np.logspace(-1, 1, 10)
accuracy = np.zeros((2, len(C_vec))) 

for i, C in enumerate(C_vec):
    clf = svm.SVC(kernel='linear', C=C)
    clf.fit(features_train, labels_train)
    #train accuracy
    pred = clf.predict(features_train)
    accuracy[0, i] = 1 - np.abs(pred-labels_train).mean()
    #test_accuracy
    pred = clf.predict(features_test)
    accuracy[1, i] = 1 - np.abs(pred-labels_test).mean()
    print(accuracy)

import matplotlib.pyplot as plt

plt.plot(C_vec, accuracy.T)
plt.show()
'''

C = 10000
gamma = 1

#clf = svm.SVC(kernel='linear', C=C)
clf = svm.SVC(kernel='rbf', gamma=gamma, C=C)
t = time()
clf.fit(features_train, labels_train)
print('train time=%.2f' % (time() - t))

t = time()
pred = clf.predict(features_test)
print('predict time=%.2f' % (time() - t))
accuracy = 1 - np.abs(pred-labels_test).mean()

print('accuracy=%.4f' % accuracy)

