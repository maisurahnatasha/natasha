import streamlit as st 
import pandas as pd
import seaborn as sns
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix 
from sklearn.metrics import classification_report 


iris = sns.load_dataset('iris') # returns a pandas dataframe
import pandas as pd


X_iris = iris.drop('species', axis=1)  
y_iris = iris['species']


xtrain, xtest, ytrain, ytest = train_test_split(X_iris, y_iris, random_state = 0)
clf = SVC(kernel='rbf', C=1).fit(xtrain, ytrain)
st.write('Iris dataset')
st.write('Accuracy of RBF SVC classifier on training set: {:.2f}'
     .format(clf.score(xtrain, ytrain)))
st.write('Accuracy of RBF SVC classifier on test set: {:.2f}'
     .format(clf.score(xtest, ytest)))
