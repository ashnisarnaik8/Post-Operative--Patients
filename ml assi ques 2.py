# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 20:08:36 2022

@author: HP
"""
import numpy as np
import pandas as pd
from numpy import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

df1 = pd.read_csv(r"C:\Users\HP\Documents\post-operative.data",header=None)
df1

df1.info()

df1.rename(columns=
               {0 : 'core_temp' ,
                1 : 'surf_temp' ,
                2 : 'o2_satur' ,
                3 : 'bp' ,
                4 : 'surf_temp_stbl' ,
                5 : 'core_temp_stbl' ,
                6 : 'bp_stbl' ,
                7 : 'comfort' ,
                8 : 'decision'},inplace=True)
df1

df1.info()

df1['core_temp'] = df1['core_temp'].replace({'high' : 2 , 'mid' : 1 ,'low' : 0})
df1['surf_temp'] = df1['surf_temp'].replace({'high' : 2 , 'mid' : 1 ,'low' : 0})
df1['bp'] = df1['bp'].replace({'high' : 2 , 'mid' : 1 ,'low' : 0})
df1['surf_temp_stbl'] = df1['surf_temp_stbl'].replace({'stable' : 2 , 'mod_stable' : 1 ,'unstable' : 0})
df1['core_temp_stbl'] = df1['core_temp_stbl'].replace({'stable' : 2 , 'mod_stable' : 1 ,'mod-stable' : 1 ,'unstable' : 0})
df1['bp_stbl'] = df1['bp'].replace({'stable' : 2 , 'mod_stable' : 1 ,'unstable' : 0})
df1['o2_satur'] = df1['o2_satur'].replace({'poor' : 0 , 'fair' : 1 , 'good' : 2 , 'excellent' : 3})
df1['decision'] = df1['decision'].replace({'I' : 0 , 'A' : 1 ,'A ' : 1, 'S' : 2})
df1

df1.info()

df1['comfort'].value_counts()
df1['comfort'] = df1['comfort'].str.replace('?' , '10').astype(int)

df1.info()

from sklearn.preprocessing import StandardScaler
 
scaler = StandardScaler()
 
scaler.fit(df1.drop('decision', axis = 1))
scaled_features = scaler.transform(df1.drop('decision', axis = 1))
scaled_features 

df1_feat = pd.DataFrame(scaled_features, columns = df1.columns[:-1])
df1_feat.head()

independent_variables = list(set(df1_feat.columns.to_list()))

X_train, X_test, y_train, y_test = train_test_split(
      scaled_features, df1['decision'], test_size = 0.25)

X_test

####################################LOGISTICREGRESSION#############################################################################

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(multi_class='multinomial',fit_intercept=True, max_iter=10000)
model.fit(X_train, y_train)
model
LogisticRegression(max_iter=10000, multi_class='multinomial')

train_preds = model.predict_proba(X_train)
train_preds
test_preds = model.predict_proba(X_test)
test_preds
train_class_preds = model.predict(X_train)
train_class_preds
test_class_preds = model.predict(X_test)
test_class_preds

table = pd.DataFrame(test_class_preds,y_test).reset_index()
table
table.columns = ['predicted', 'actual']
table

from sklearn.metrics import accuracy_score, confusion_matrix
train_accuracy = accuracy_score(train_class_preds,y_train)
train_accuracy
test_accuracy = accuracy_score(test_class_preds,y_test)
test_accuracy

print("The accuracy on train data is ", train_accuracy)
print("The accuracy on test data is ", test_accuracy)

cm = confusion_matrix(y_test, test_class_preds)
print(cm)

labels = ['I' , 'A' , 'S']
ax= plt.subplot()
sns.heatmap(cm,square = True,annot = True)
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(labels)
ax.yaxis.set_ticklabels(labels)

#############################RANDOMFORESTCLASSIFIER########################################################################################################

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
from pylab import rcParams
rcParams['figure.figsize'] = 8, 8

classifier = RandomForestClassifier(n_estimators=100)
classifier.fit(X_train, y_train)

RandomForestClassifier()
test_preds_  = classifier.predict(X_test)
test_preds
train_preds_ = classifier.predict(X_train)
train_preds

accuracy_score(y_test , test_preds_)

confusion_matrix(y_test , test_preds_)
feature_imp = pd.Series(classifier.feature_importances_, index = independent_variables).sort_values(ascending = False)
feature_imp

####################################SVM############################################################################################

from sklearn.svm import SVC
SVC_model = SVC().fit(X_train , y_train)
SVC_model

test_p = SVC_model.predict(X_test)
test_p
accuracy_score(test_p , y_test)

#################################KNN####################################################################################

from sklearn.neighbors import KNeighborsClassifier
 
knn = KNeighborsClassifier(n_neighbors = 13)
 
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
pred

accuracy_score(pred , y_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, pred))
 
print(classification_report(y_test, pred))

neighbors = np.arange(1, 20)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))
  
for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
        
    train_accuracy[i] = knn.score(X_train, y_train)
    test_accuracy[i] = knn.score(X_test, y_test)
  
plt.plot(neighbors, test_accuracy, label = 'Testing dataset Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training dataset Accuracy')
  
plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.show()

from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=4).fit(X_train, y_train)
tst_p = knn_model.predict(X_test)
accuracy_score(y_test , tst_p)
