# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 09:09:44 2018

@author: 5-310
"""

import numpy as np

"""
한 세대에서 태어나는 아이의 수는 3으로 한다.
아이의 성별의 비율은 5:5로 한다.
아이의 성은 아버지(남자)를 따른다.
"""

np.random.uniform()

np.random.normal()

np.round(np.random.uniform()) ## 반올림 함수(유니폼 분포 난수)

np.sign(np.random.normal())  ## 반올림 함수(정규분포 난수)


child = 0
for _ in np.arange(3):
    child += np.round(np.random.uniform()) ## 1이면 아들 0이면 딸 (child = 아들)
    
parent = child
child = 0
for _ in np.arange(parent):
    for _ in np.arange(3):
        child += np.round(np.random.uniform())
print('parent=%d'%parent)
print('child=%d'%child)


def child(n):
    res = 0
    for _ in np.arange(n):
        for _ in np.arange(3): ## 처음에 한 커플당 아이 3명씩 낳는다 가정했음
            res += np.round(np.random.uniform())
    return(res)
    
    
n = 1
gene = 1
while(True):
    init = child(n) ## 최초의 아이(아들) 값
    if (init == 0 or gene == 10):
        break  ## stop 문장 꼭 넣자!!
    n = init
    gene += 1
    print(init,gene)
    
## 역행렬
np.linalg.inv
## 행렬 곱
np.dot()
## (1,x)행렬
np.vstack(); np.hstack()

### 기계학습 예제 (내가 푼 것, 실패)
import matplotlib as mpl
import matplotlib.pylab as plt

x = [11,19,23,26,29,30,38,39,46,49]
y = [29,33,51,40,49,50,69,70,64,89]
x = np.array([x]).T
y = np.array([y]).T
plt.plot(x, y)

import pandas as pd

b = [(x),(y)]

df = pd.DataFrame([b]).T
df.corr()


### 기계학습 예제 (강사님 풀이)

x = [11,19,23,26,29,30,38,39,46,49]
y = [29,33,51,40,49,50,69,70,64,89]
x = np.array(x)
y = np.array([y]).T
x1 = np.vstack([np.ones(len(x)),x]).T

x_inv = np.linalg.inv(np.dot(x1.T,x1))
xy = np.dot(x1.T,y)
beta = np.dot(x_inv,xy)
beta[0] + beta[1]*x ## y 예측 값
y ## 실제 y값

## 함수로 만들기 (**list 형태만 가능!!!)
def reg(x,y):
    x = np.array(x)
    y = np.array([y]).T
    x1 = np.vstack([np.ones(len(x)),x]).T
    x_inv = np.linalg.inv(np.dot(x1.T,x1))
    xy = np.dot(x1.T,y)
    beta = np.dot(x_inv,xy)
    return(beta)

reg(x,y)

## 레드와인 퀄리티 예측
redwine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv', delimiter=';')

redwine.corr( )
redwine[['sulphates','density','total sulfur dioxide','free sulfur dioxide','chlorides','residual sugar','citric acid','fixed acidity','volatile acidity','alcohol','quality','pH']].corr( )

redwine_group = redwine['alcohol'].groupby(redwine['quality'])
redwine_group.mean( )
redwine_group.std( )

reg(redwine['alcohol'],redwine['quality'])

reg(redwine['quality'],redwine['alcohol'])

## 강사님 풀이
redwine2 = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', delimiter=';')
type(redwine2)
redwine2[['quality']] ## 여러가지 컬럼 불러오기 가능
redwine2['quality'] ## 한가지 컬럼만 가능
redwine2.columns

list(redwine2[['quality']])
list(redwine2['quality'])
a = list(redwine2[['quality']].values.flatten()) ## .values 왜 쓰는지??

np.array([a]).T

## 사실상 여기부터 시작
y = redwine2[['quality']]
x = redwine2[redwine2.columns[0:-1]]

## 오류 케이스 np.hstack(np.array([np.ones(len(x))]).T,x)

x1 = np.hstack([np.array([np.ones(len(x))]).T,x])

""" ## 풀어쓰면 이럼
one_array = np.ones(len(x))
one_matrix = np.array([one_array])
x1 = np.hstack([one_matrix.T,x])
"""

x_inv = np.linalg.inv(np.dot(x1.T,x1))
xy = np.dot(x1.T,y)
beta = np.dot(x_inv,xy)

np.dot(x1,beta)

y_hat = np.round(np.dot(x1,beta))
sum(y.values != y_hat)/len(y_hat)
sum(y.values == y_hat)/len(y_hat)
beta
redwine2.columns[0:-1]
pd.Series(beta[1:].flatten(),
          index=redwine2.columns[0:-1])

x[['density']].mean()
x.mean()
x.std() ## 표준편차
## density의 표준편차가 매우작다.
"""
변수     대략적인 분포   영향도
density  1.001~0.990     -17
alcohol  11.5~9.3        0.27
"""

a = [True,True,True,False,False]
sum(a)

## 양키 풀이 보고 내가 풀어본 것
redwine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', delimiter=';')

redwine.corr( )
redwine[['sulphates','density','total sulfur dioxide','free sulfur dioxide','chlorides','residual sugar','citric acid','fixed acidity','volatile acidity','alcohol','quality','pH']].corr( )

redwine['alcohol']
redwine['quality']

alc = np.hstack(redwine['alcohol'])
qua = np.hstack(redwine['quality'])

beta = reg(alc,qua)

beta[0]+beta[1]*alc

###끝

housing = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
            ,delim_whitespace = True,
            header = 0) ## 칼럼 이름이 비어있어서 헤더 = 0으로 한다는데..
## 칼럼 이름 넣기
hou_name = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']

housing.columns = hou_name

y = housing[['MEDV']]
x = housing[housing.columns[0:-1]]
x1 = np.hstack([np.array([np.ones(len(x))]).T,x]) ## 변형
x_inv = np.linalg.inv(np.dot(x1.T,x1)) ## 계산식에 따른 절차
xy = np.dot(x1.T,y) ## 계산식에 따른 절차
beta = np.dot(x_inv,xy) ## 계산식에 따른 절차
y_hat = np.round(np.dot(x1,beta)) ## 예측 값
hou_beta = pd.Series(beta[1:].flatten(),
          index=housing.columns)
err = np.sqrt(np.sum((y - y_hat)**2))

crim = housing['CRIM']
medv = housing['MEDV']

import matplotlib.pylab as plt
plt.plot(crim,medv,'ro')
plt.plot(crim,crim*hou_beta['CRIM'])


hou_beta['CRIM']







