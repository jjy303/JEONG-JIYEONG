# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 09:05:15 2018

@author: 5-310
"""

import sklearn
from sklearn.model_selection import train_test_split

sklearn.model_selection.train_test_split ## 이 함수를 통해 데이터를 나눠보자

import pandas as pd
import numpy as np

housing = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
            ,delim_whitespace = True,
            header = 0) ## 칼럼 이름이 비어있어서 헤더 = 0으로 한다는데..

hou_name = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']

housing.columns = hou_name

len(housing.columns)
    
    X = housing[housing.columns[0:-1]] ## 집 값을 뺀 것
    Y = housing[['MEDV']] ## 집 값
    

betaT = []
errT = []    
for _ in np.arange(1000):
    ## 데이터마이닝 !!
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y)
    
    X_train = np.hstack([np.array([np.ones(len(X_train))]).T,X_train])
    x_inv = np.linalg.inv(np.dot(X_train.T,X_train))
    xy = np.dot(X_train.T,Y_train)
    beta = np.dot(x_inv,xy)  ## 훈련데이터 모형
    
    X_test = np.hstack([np.array([np.ones(len(X_test))]).T,X_test])
    y_hat = np.round(np.dot(X_test,beta))
    err = np.sqrt(np.sum((Y_test - y_hat)**2))  ## 실험데이터 모형
    
    errT.append(err)
    betaT.append(beta)

np.where(np.min(errT)==errT)[0]
betaT[int(np.where(np.min(errT)==errT)[0])]
np.min(errT)



## 와인 데이터로 실습해보기 !!

redwine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', delimiter=';')
X = redwine[redwine.columns[0:-1]]
Y = redwine[['quality']]

betaT = []
errT = []    
err1T = []
err2T = []
for _ in np.arange(10): ## 데이터마이닝 !!
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y)
    
    X_train = np.hstack([np.array([np.ones(len(X_train))]).T,X_train])
    x_inv = np.linalg.inv(np.dot(X_train.T,X_train))
    xy = np.dot(X_train.T,Y_train)
    beta = np.dot(x_inv,xy)
    
    X_test = np.hstack([np.array([np.ones(len(X_test))]).T,X_test])
    y_hat = np.round(np.dot(X_test,beta))

    err = sum(Y_test.values != y_hat)/len(y_hat)
    err1 = sum(Y_test.values != y_hat)
    err2 = sum(Y_test.values == y_hat)
    
    errT.append(err)
    betaT.append(beta)
    err1T.append(err1)
    err2T.append(err2)

np.where(np.min(errT)==errT)[0]
betaT[int(np.where(np.min(errT)==errT)[0])]
print(np.min(errT),np.min(err1T),np.max(err2T))

Y_hat = np.round(np.dot(X_train,beta)) ## SSR SST R^2 구하기
sst = np.sum((Y - np.mean(Y))**2)
ssr = np.sum((Y_hat - np.mean(Y).values)**2)

R_2 = ssr/sst
R_2

## 끝 !

## Gradient Descent 알고리즘 << 작동을 안한다.. 다른 데이터로 다시 해보자

housing = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
            ,delim_whitespace = True,
            header = 0) ## 칼럼 이름이 비어있어서 헤더 = 0으로 한다는데..

hou_name = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']

housing.columns = hou_name

x = housing[housing.columns[0:-1]] ## 집 값을 뺀 것
y = housing[['MEDV']] ## 집 값
x1 = np.hstack([np.array([np.ones(len(x))]).T,x])

m = len(x1)
beta = np.zeros([x1.shape[1],1])
alpha = 0.000001 ## 설정 이동거리

for _ in range(1000):
    y_hat = np.dot(x1,beta)
    loss = y_hat - y
    cost = np.sum(loss**2)/(2*m)
    gradient = np.dot(x1.T,loss)/m
    beta = beta - alpha * gradient ## 베타 업데이트


###

import sklearn
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
df = pd.read_csv("https://raw.githubusercontent.com/rasbt/python-machine-learning-book/master/code/datasets/housing/housing.data", delimiter=r"\s+", 
                names = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"])

df.head() ## ()안에 숫자를 넣으면 상위부터 설정 숫자만큼만 = df.tail()과 같다
df.shape
df[pd.isnull(df).any(axis=1)] ## 각 row별로 null값이 있었는지 검사 *any함수는 어느 하나라고 True이면 True

X = df[["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]]
y = df[["MEDV"]]


## 선형회귀분석


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=5)
reg = linear_model.LinearRegression() ## class의 변수 선언(?) 
reg.fit(X_train, y_train) ## 모형 fit , '모형을 만든다' -> 훈련 데이터(train)
reg.intercept_ ## 적합된 데이터의 intercept(?..계수?)를 확인
reg.coef_ ## 코피션트 계수
reg.predict([[0.03237, 0.0, 2.18, 0, 0.458, 6.998, 45.8, 6.0622, 3, 222.0, 18.7, 394.63, 2.94]]) ## []안의 값을 사용한 예측값을 보여줌
y_pred=reg.predict(X_test) ## 이러한 데이터베이스 형태로 넣어도 되고, 두개의 list로 넣어도 된다(바로 위줄에 있음)
y_pred[0:5]
y_test_m = y_test.as_matrix()


## 그림


plt.figure(figsize=(15,10))
plt.plot(y_test_m, ms=50, alpha=1) ## y_test는 데이터베이스 형태인데, _m을 붙이지 않으면 크기순으로 정리하지 않는다!!
plt.plot(y_pred, ms=50, alpha=2)
legend_list = ['y_test_m', 'y_pred']
plt.legend(legend_list, loc=4, fontsize='20')

mean_squared_error(y_test, y_pred) ## 
# 20.869292183770256

r2_score(y_test, y_pred) ## 결정계수, 인과관계가 전혀 없는 변수라도 일단 들어가기만 하면 커진다
# 0.73344921474531466

linear_model.LinearRegression ## linear_model의 가장 기본적인 것임


## 분류

import sklearn
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns ## 이미지..?
%matplotlib inline
from IPython.display import Image
from sklearn import preprocessing ## 데이터 핸들링을 위한 패키지
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics

df = pd.read_table("https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data", sep=',',
                  names = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "eval"])

df.shape
df[pd.isnull(df).any(axis=1)]
df['eval'].value_counts()
car_counts = pd.DataFrame(df['eval'].value_counts())
car_counts['Percentage'] = car_counts['eval']/car_counts.sum()[0]

## 피규어

plt.figure(figsize=(8,8))
plt.pie(car_counts["Percentage"],
       labels = ['Unacceptable','Acceptable', 'Good', 'Very Good'])
df.head()

le = preprocessing.LabelEncoder()
encoded_buying = le.fit(df['buying'])
encoded_buying.classes_

encoded_buying.transform(['high'])
encoded_buying.transform(['low'])
encoded_buying.transform(['med'])
encoded_buying.transform(['vhigh'])
encoded_buying.inverse_transform(1)

for i in range (0,4):
    print(i, ":", encoded_buying.inverse_transform(i))

## 바꾸고싶은 대상을 바꾸기 문자를 숫자로 숫자를 문자로

df['e.buying'] = df['buying'].map(lambda x: encoded_buying.transform([x]))
df.head()
df['e.buying'] = df['e.buying'].map(lambda x: x[0])
df.head()

encoded_buying.transform(['low'])

##

encoded_maint = le.fit(df['maint'])
encoded_maint.classes_
df['e.maint'] = df['maint'].map(lambda x: encoded_maint.transform([x]))
df['e.maint'] = df['e.maint'].map(lambda x: x[0])
df.head()


## 함수로 만들어서 쓰기

def encode_col(col_name):
    encodes = le.fit(df[col_name])
    new_col_name = "e."+col_name
    df[new_col_name] = df[col_name].map(lambda x: encodes.transform([x]))
    df[new_col_name] = df[new_col_name].map(lambda x: x[0])
    return 

##
'''
encode_col('doors')
encode_col('persons')
encode_col('lug_boot')
encode_col('safety')
encode_col('eval')
df.head()
'''
df.head() ## ''' 내용을 간단하게
for i in df.columns:
    encode_col(i)
##
    
