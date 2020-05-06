# -*- coding: utf-8 -*-
"""
Created on Tue May  5 12:50:28 2020

@author: kaushik
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline



df = pd.read_csv('brazil houses_to_rent_v2.csv')


df.head() 


sum=df.iloc[:,8] + df.iloc[:,9] + df.iloc[:,10] + df.iloc[:,11]

sel_col=df.iloc[:,[8,9,10,11]]

rent=df.iloc[:,-1]

from sklearn.model_selection import train_test_split
X_train_, X_test_, y_train_, y_test_ = train_test_split(sel_col,rent,test_size=.25, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


X_train_ = sc.fit_transform(X_train_)
X_test_ = sc.transform(X_test_)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()


lr.fit(X_train_,y_train_)

lr.score(X_test_,y_test_)


lr.score(X_train_,y_train_)

pd.DataFrame({'Actual':list(y_test_), 'predicted': list(lr.predict(X_test_))})


from sklearn.linear_model import LinearRegression

reg = LinearRegression()

X = df.iloc[:,[8,9,10,11]]

y= df.iloc[:,[12]]

reg.fit(X,y)

reg_pred = reg.predict(X)


print(reg_pred)


import pickle

filename='Brazil_Rent.pkl'
pickle.dump(reg,open(filename,'wb'))


model = pickle.load(open('Brazil_Rent.pkl','rb'))


#np.array([[2065,3300,211,42]]).reshape(-1,1)


#model.predict([[2065,3300,211,42]])
