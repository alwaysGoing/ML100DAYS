# -*- coding: utf-8 -*-
# @Time    : 2019/7/18 14:41
# @Author  : Wang
# @FileName: Data_preprocessing.py
# @Software: PyCharm

import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

# step 2
print('----------step2----------')
dataset = pd.read_csv('../Dataset/Data.csv')
print(dataset)
# print(type(dataset))
# 第一列为行号，不计入shape
print(dataset.shape)
print(type(dataset))
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values
print(X)
print(X.shape)
print(type(X))
print(Y)

# step 3
# 处理缺失值
print('---------step3----------')
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
print(X)
print(type(X))

# step 4
print('---------step4---------')
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
print(X)
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

print(X)
print(X.shape)
print(Y)

# step 5
print('----------step5----------')
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
print(X_train)
print(X_test)
print(Y_train)
print(Y_test)

# step 6, 特征量化
print('----------step6----------')
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
print(X_train)
print(X_test)
