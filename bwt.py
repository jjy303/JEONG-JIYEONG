# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 12:54:33 2018

@author: 5-310
"""


import sklearn
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns ## 이미지..?
from IPython.display import Image
from sklearn import preprocessing ## 데이터 핸들링을 위한 패키지
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

tt = pd.read_csv('C:/Users/5-310/Desktop/samples/reg_level2t.csv', delimiter=',')

tt.head() ## 데이터의 전체적인 확인

tt.bwt.value_counts() ## 분석대상이 어떻게 이루어져 있는지 확인

np.unique(tt.bwt) ## 대상 어레이의 중복값을 제외한 것을 보여주는 함수


tt.shape
## tt[pd.isnull(tt).any(axis=1)]

tt = tt.drop(tt.columns[0:1],axis = 1)


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


## 데이터 설정
X = tt[tt.columns[0:-1]]
y = tt['bwt']
X_train, X_test, y_train, y_test = train_test_split(X,y)

X.head()
y.head()

## 모형 적합 (모형 만드는 과정)
'''
solver = 'newton-cg'
model = 'multinomial'
lr = LogisticRegression(solver=solver,
                        multi_class=model,
                        C=1,
                        penalty='l2',
                        )
logistic -> 연속형 변수에 적합하지 않다 ! -> 이걸 쓰면 결정계수는 안보는거다 생각 !
'''
lr = linear_model.LinearRegression()
lr.fit(X_train,y_train)

## 오차 확인
y_pred = lr.predict(X_test)
s = r2_score(y_test, y_pred)
## >> 확률이 낮으면 위의 설정을 계속 바꿔서 시도해 볼 것

X.head()
X.mean()
X.std() ## std = 표준편차

mse = mean_squared_error(y_test, y_pred)

print('결정 계수 = %g\nMSE = %d'%(s,mse))

y_test_m = y_test.as_matrix()

plt.figure(figsize=(15,10))
plt.plot(y_test_m, ms=50, alpha=1)
plt.plot(y_pred, ms=50, alpha=2)
legend_list = ['y_test_m','y_pred']
plt.legend(legend_list, loc=4, fontsize='25')

