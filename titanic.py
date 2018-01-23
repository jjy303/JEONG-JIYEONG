# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 09:05:29 2018

@author: 5-310
"""

'''
df[pd.isnull(df).any(axis=1)]
비어있는 데이터는 쓰지 않는다 (널값 제거하기)
'''

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

tt = pd.read_csv('C:/Users/5-310/Desktop/samples/cf_level1t.csv', delimiter=',')

tt.head() ## 데이터의 전체적인 확인

tt.survived.value_counts() ## 분석대상이 어떻게 이루어져 있는지 확인

np.unique(tt.survived) ## 대상 어레이의 중복값을 제외한 것을 보여주는 함수


tt.shape
## tt[pd.isnull(tt).any(axis=1)]
tt['survived'].value_counts()
survived_counts = pd.DataFrame(tt['survived'].value_counts())
survived_counts['Percentage'] = survived_counts['survived']/survived_counts.sum()[0]


plt.figure(figsize=(8,8))
plt.pie(survived_counts["Percentage"],
       labels = ['dead','survived'])
tt.head() ## 그래프를 통해 간단히 확인


tt = tt.drop(['name','ticket','cabin','boat','body','home.dest'],axis = 1)

##idx = tt['embarked'].dropna(axis=0).index ## 임바키드 널값 제거 !! 안쓰지만, 다음에 필요할지도 모름
##tt = tt.ix[idx]

tt = tt.dropna(axis = 0)


le = preprocessing.LabelEncoder()

def encode_col(col_name):
    encodes = le.fit(tt[col_name])
    new_col_name = "e."+col_name
    tt[new_col_name] = tt[col_name].map(lambda x: encodes.transform([x]))
    tt[new_col_name] = tt[new_col_name].map(lambda x: x[0])
    return 

encode_col('sex')
encode_col('embarked')
tt.head()


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

solver = 'newton-cg'
model = 'multinomial'
lr = LogisticRegression(solver=solver,
                        multi_class=model,
                        C=1,
                        penalty='l2',
                        )

## 데이터 설정
X = tt[['pclass','e.sex','age','sibsp','parch','fare','e.embarked']]
y = tt['survived']
X_train, X_test, y_train, y_test = train_test_split(X,y)


## 모형 적합 (모형 만드는 과정)
lr.fit(X_train,y_train)

## 오차 확인
y_pred = lr.predict(X_test)
s = np.sum(y_pred == y_test) / y_test.shape[0] ## y_test.shape[0] = 로우와 컬럼 숫자를 어레이 형태로 ,, 맞은거의 갯수를 전체 갯수로 나눈 것, 즉 맞은 확률
## >> 확률이 낮으면 위의 설정을 계속 바꿔서 시도해 볼 것

X.head()
X.mean()
X.std() ## std = 표준편차

print('예측 성공 확률 = %g'%s)