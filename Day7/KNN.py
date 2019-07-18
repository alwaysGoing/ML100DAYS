# -*- coding: utf-8 -*-
# @Time    : 2019/7/18 20:33
# @Author  : Wang
# @FileName: KNN.py
# @Software: PyCharm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# step 1
print('----------step1----------')
dataset = pd.read_csv('../Dataset/Social_Network_Ads.csv')
print(dataset[:10])

X = dataset.iloc[:, 2:-1].values
Y = dataset.iloc[:, -1].values

# step 2
print('----------step4----------')
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# step 3
print('----------step3----------')
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# step 4
print('----------step4----------')
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X_train, Y_train)

# step 5
print('----------step5----------')
y_pred = classifier.predict(X_test)
print(y_pred)

# step 6
print('----------step6----------')
cm = confusion_matrix(Y_test, y_pred)
print(cm)

