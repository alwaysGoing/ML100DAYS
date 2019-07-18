# -*- coding: utf-8 -*-
# @Time    : 2019/7/18 16:58
# @Author  : Wang
# @FileName: MultipleLinearRegression.py
# @Software: PyCharm

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

# step 1, 导入数据
print('----------step1----------')
dataset = pd.read_csv('../Dataset/50_Startups.csv')
print(dataset)
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values
print(X[:10])
print(Y)

# onehot编码
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
print(X[:10])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()
print(X[:10])

# 躲避虚拟变量陷阱
X1 = X[:, 1:]

# 拆分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y, test_size=0.2, random_state=0)

# step 2, 在训练集上训练多元线性回归模型
print('---------step2----------')
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
regrassor1 = LinearRegression()
regrassor1.fit(X1_train, Y1_train)

# step 3, 在测试集上预测结果
print('----------step3----------')
y_pred = regressor.predict(X_test)
y1_pred = regrassor1.predict(X1_test)

print(y_pred)
print(y1_pred)
