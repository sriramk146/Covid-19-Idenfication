import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import tkinter
from tkinter import *
from tkinter import messagebox
tkWindow = Tk()
tkWindow.geometry('400x150')
tkWindow.title('Tkinter Example')
def SVM():
 # Importing the dataset
 dataset = pd.read_excel('')
 X = dataset.iloc[, ].values
 y = dataset.iloc[, ].values
 from sklearn.model_selection import train_test_split
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,shuffle=False)
 from sklearn import svm
 #Create a svm Classifier
 clf = svm.SVC(kernel='linear') # Linear Kernel
 #Train the model using the training sets
 clf.fit(X_train, y_train)
 #Predict the response for test dataset
 y_pred = clf.predict(X_test)
 #Import scikit-learn metrics module for accuracy calculation
 from sklearn import metrics

 # Model Accuracy: how often is the classifier correct?
 print("Accuracy for svm:",metrics.accuracy_score(y_test, y_pred))
 print(confusion_matrix(y_test, y_pred))
 print(classification_report(y_test, y_pred))
button = tkinter.Button(tkWindow, text='SVM', command=SVM)
button.pack()
def NB():
 # Importing the dataset
 dataset = pd.read_excel('')
 X = dataset.iloc[,].values
 y= dataset.iloc[,].values
 from sklearn.model_selection import train_test_split
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,shuffle=True)
 from sklearn.naive_bayes import GaussianNB
 #Create a Gaussian Classifier
 gnb = GaussianNB()
 #Train the model using the training sets
 gnb.fit(X_train, y_train)
 #Predict the response for test dataset
 y_pred = gnb.predict(X_test)
 from sklearn import metrics
 # Model Accuracy, how often is the classifier correct?
 print("Accuracy for naive bayes:",metrics.accuracy_score(y_test, y_pred))
button1 = tkinter.Button(tkWindow, text='NB', command=NB)
button1.pack()
def DV():
 dataset = pd.read_excel('')
 X = dataset.iloc[,].values
 y = dataset.iloc[,].values
 y1 = dataset.iloc[,].values
 X1 = dataset.iloc[,].values
 X2 = dataset.iloc[,].values
 X3 = dataset.iloc[,].values
 plt.plot(y1,X1,c='orange',label='Fever')
 plt.plot(y1,X2,c='red',label='Breadth rate')
 plt.plot(y1,X3,c='blue',label='Hyperthermia')
 plt.title('Sensor Data Visualization')
 plt.xlabel('No. of Person')
 plt.ylabel('Sensor Data')
 plt.legend()
 plt.show()
button2 = tkinter.Button(tkWindow, text='Data Visualization', command=DV)
button2.pack()
tkWindow.mainloop()
