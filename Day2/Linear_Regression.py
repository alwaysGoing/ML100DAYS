# -*- coding: utf-8 -*-
# @Time    : 2019/7/18 16:31
# @Author  : Wang
# @FileName: Linear_Regression.py
# @Software: PyCharm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

# step 1, 数据预处理
print('----------step1----------')
dataset = pd.read_csv('../Dataset/studentscores.csv')
print(dataset)

X = dataset.iloc[:, :1].values
Y = dataset.iloc[:, 1].values
print(X)
print(type(X))  # ndarray
print(Y)
print(type(Y))  # ndarray

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/4, random_state=0)

# step 2, 训练线性回归
print('----------step2----------')
regressor = LinearRegression()
regressor = regressor.fit(X_train, Y_train)

# step 3, 预测结果
print('----------step3----------')
Y_pred = regressor.predict(X_test)

# step 4, 可视化
print('----------step4----------')
print('散点图')
plt.scatter(X_train, Y_train, color='red')
print('线图')
plt.plot(X_train, regressor.predict(X_train), 'bo-')
plt.show()

# 测试集结果可视化
plt.scatter(X_test, Y_test, color='red')
plt.plot(X_test, Y_pred, 'bo-')
plt.show()

