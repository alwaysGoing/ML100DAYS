# -*- coding: utf-8 -*-
# @Time    : 2019/7/18 18:33
# @Author  : Wang
# @FileName: Logistic_regression.py
# @Software: PyCharm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap

# step 1
print('----------step1----------')
dataset = pd.read_csv('../Dataset/Social_Network_Ads.csv')
X = dataset.iloc[:, 2:-1].values
Y = dataset.iloc[:, -1].values
print(dataset[:10])
print(X[:10])
print(Y[:10])

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.25, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# step 2
print('----------step2----------')
classifier = LogisticRegression()
classifier.fit(X_train, Y_train)

# step 3
print('----------step3----------')
y_pred = classifier.predict(X_test)

# step 4
print('----------step4----------')
# 生成混淆矩阵
cm = confusion_matrix(Y_test, y_pred)

# 可视化
# X_set, y_set = X_train, Y_train
# X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() -
#                                1, stop=X_set[:, 0].max() +
#                                1, step=0.01), np.arange(start=X_set[:, 1].min() -
#                                                         1, stop=X_set[:, 1].max() +
#                                                         1, step=0.01))
# plt.contour(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(
#     X1.shape), alpha=0.75, cmap=ListedColormap(('red', 'green')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
X_set, y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
print(X1.shape)
print(X2.shape)
a = classifier.predict(np.array([X1.ravel(), X2.ravel()]).T)
print(a[:10])
print(a.shape)
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)

plt.title(' LOGISTIC(Training set)')
plt.xlabel(' Age')
plt.ylabel(' Estimated Salary')
plt.legend()
plt.show()

X_set, y_set = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() -
                                     1, stop=X_set[:, 0].max() +
                                             1, step=0.01), np.arange(start=X_set[:, 1].min() -
                                                                            1, stop=X_set[:, 1].max() +
                                                                                    1, step=0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(
    X1.shape), alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)

plt.title(' LOGISTIC(Test set)')
plt.xlabel(' Age')
plt.ylabel(' Estimated Salary')
plt.legend()
plt.show()
