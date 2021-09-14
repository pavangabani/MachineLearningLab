# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 14:19:52 2021

@author: Hp
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns ;sns.set()
from sklearn.model_selection import train_test_split


data=pd.read_csv("titanic.csv")
print(data.head())
print(data.describe())
data.drop(['Name'],axis=1)

survived_count=data.groupby('Survived')['Survived'].count()

plt.figure(figsize=(4,5))
plt.bar(survived_count.index, survived_count.values)
plt.title('Grouped by survival')
plt.xticks([0,1],['Not survived', 'Survived'])
for i, value in enumerate(survived_count.values):
    plt.text(i, value-70, str(value), fontsize=12, color='white',
             horizontalalignment='center', verticalalignment='center')
plt.show()

predictor=data.drop(['Survived'],axis=1)
target=data['Survived']

x_train,x_test,y_train,y_test=train_test_split(predictor,target ,test_size=0.2)

from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)

from sklearn.metrix import accuracy_score
print(accuracy_score(y_pred, y_test))

