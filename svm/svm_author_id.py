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


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
from sklearn.svm import SVC
import numpy as np

model = SVC(kernel='rbf', C=10000)
t0 = time()
#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]
model.fit(features_train, labels_train)
print ("training time:", round(time()-t0, 3), "s")

t0 = time()
print model.score(features_test, labels_test)
print ("testing time:", round(time()-t0, 3), "s")

predictions = model.predict(features_test)
print predictions[10]
print predictions[26]
print predictions[50]

print np.count_nonzero(predictions == 1)
#########################################################