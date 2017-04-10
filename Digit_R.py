#Auto Generated Code Portion by IDE(Spyder)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#|------------------------------------------|
#| Authors @ Anshul Chauhan (30/EC/14)      |
#|         @ Anmol Chandra Singh(29/EC/14)  |
#|         @ Amish Singh (22/EC/14)         |
#| EC - 319 Project                         |
#| Title : Digit Recognition                |
#|------------------------------------------|
#

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

#           Data import                 #
dataset=pd.read_csv('train.csv')  # Reading train.csv file
X=dataset.iloc[0:1000,1:].values  # Reading limited values from train.csv
y=dataset.iloc[0:1000,0].values
print (dataset.info())
              
#           Splitting the data into test and train set         #
from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test=train_test_split(X,              #Creates the test
                                               y,              # and train arrays
                                               test_size=0.20, # for each input
                                               random_state=0) # array
                                                               # and splits them
                                                               # into arrays
                                                               

 #           Visualising Data            #
#for i in range(100):
i = 1
print i
img=np.asmatrix(X_train[i])
img=img.reshape((28,28))
plt.imshow(img,cmap='gray')
plt.title(y_train[i])

#           Feature Scaling             #
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()                         # Scaling the values,i.e, 
X_train = sc_X.fit_transform(X_train)           # making zero mean and dividing
X_test = sc_X.transform(X_test)                 # by std deviation

#Fitting Classifier to the training set

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf',random_state=0)   # Initialing the classifier
                                                  # Using (gaussian) radial basis 
                                                  # function kernel
classifier.fit(X_train,y_train)                   # Fitting the training sets
                                                  # into the classifier

#Predicting on X_test
y_pred = classifier.predict(X_test)               # Using the classifier to
                                                  # predict the correct values
                                                  
y_score = classifier.score(X_test,y_test)         # Finding the mean of 
                                                  # accuracies

# Making the confusion matrix(for judging our result)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test , y_pred)

print y_pred
print y_score
print y_test

from sklearn.metrics import accuracy_score
print accuracy_score(y_test, y_pred)