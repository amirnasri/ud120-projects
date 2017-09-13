#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn import svm
#clf = GaussianNB()
#clf = BernoulliNB()
clf = svm.SVC()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
'''
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
from sklearn.metrics import precision_score, recall_score, f1_score
precision = precision_score(labels_test, pred)
recall = recall_score(labels_test, pred)
f1_score = f1_score(labels_test, pred)
acc = clf.score(features_test, labels_test)
'''
from sklearn.cross_validation import StratifiedShuffleSplit
import numpy as np

labels = np.array(labels)
features = np.array(features)
n_splits = 1000
sss = StratifiedShuffleSplit(y=labels, n_iter=n_splits, test_size=0.1)
n_total = 0
tp, tn, fp, fn = 0, 0, 0, 0
for train_index, test_index in sss:
    features_train, labels_train = features[train_index], labels[train_index]
    features_test, labels_test = features[test_index], labels[test_index]

    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)

    tp += np.sum((pred == 1) & (labels_test == 1))
    tn += np.sum((pred == 0) & (labels_test == 0))
    fp += np.sum((pred == 1) & (labels_test == 0))
    fn += np.sum((pred == 0) & (labels_test == 1))
    n_total += len(labels_test)

acc = float(tp+tn)/n_total
precision = float(tp)/(tp+fp)
recall = float(tp)/(tp+fn)
f1_score = 2 * precision*recall/(precision + recall)
print('acc=%.4f precision=%.4f recall=%.4f f1=%.4f' % (acc, precision, recall, f1_score))



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
