# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 09:23:13 2018

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

df = pd.read_table("https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data", sep=',',
                  names = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "eval"])

df.shape
df[pd.isnull(df).any(axis=1)]
df['eval'].value_counts()
car_counts = pd.DataFrame(df['eval'].value_counts())
car_counts['Percentage'] = car_counts['eval']/car_counts.sum()[0]


plt.figure(figsize=(8,8))
plt.pie(car_counts["Percentage"],
       labels = ['Unacceptable','Acceptable', 'Good', 'Very Good'])
df.head()

le = preprocessing.LabelEncoder()
encoded_buying = le.fit(df['buying'])
encoded_buying.classes_

encoded_buying.transform(['high']) ## 확인용 !!
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
    
    
pd.DataFrame(df['eval'].value_counts())
pd.DataFrame(df['e.eval'].value_counts())

X = df[['e.buying', 'e.maint', 'e.doors', 'e.persons', 'e.lug_boot', 'e.safety']]
type(X)
X.shape

y = df['e.eval']
type(y)
y.shape
len(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)
    
X_train.shape
y_train.shape
X_test.shape
y_test.shape

#Decision Trees를 이용
clf_dt = tree.DecisionTreeClassifier(random_state=10) #####################################
clf_dt.fit(X_train, y_train)                          ### 분석하는 명령은 단 3줄밖에 안된다.!!
y_pred_dt = clf_dt.predict(X_test)                    ###### y_pred_dt = 예측한 결과########
type(y_pred_dt)
y_pred_dt.shape
y_pred_dt


print(metrics.accuracy_score(y_test, y_pred_dt))
np.sum(y_test==y_pred_dt)/len(y_test)

correct_pred_dt = []
wrong_pred_dt = []

y_test2 = y_test.reset_index(drop = True)
y_test2 = y_test2.as_matrix()
for i in range(len(y_test2)):
    if y_test2[i] != y_pred_dt[i]:
        wrong_pred_dt.append(i)
    else:
        correct_pred_dt.append(i)
  
    
print("Correctly indetified labels: ", len(correct_pred_dt))
print(" ")
print("Wrong indetified labels: ", len(wrong_pred_dt)) ## 참고로, 프린트 함수는 프로그램 실행 시간이 길 때, 얼마나 완료되었는지 확인용으로도 사용 가능하다

X_test.head()
y_test[0:5]
y_pred_dt[0:5]


X_test.ix[2]    ####
index_num = 2   #### test 용 설정
def dt_probs(index_num):
    X_param = X_test.ix[index_num]
    X_param = X_param.to_frame()
    X_param = X_param.transpose()
    temp_pred = clf_dt.predict_proba(X_param) ## 예측에 대한 확률 [ 0 0 1 0 ] -> 언어셉터블 어셉터블 굿 베리굿 순서로 확률
    temp_pred_1 = temp_pred[0]
    y_actual = y_test[index_num]
    y_range = ['Unacceptable','Acceptable', 'Good', 'Very Good']
    print("For index number: ", index_num)
    print(" ")
    print("Fetures entered: ")
    print(X_param)
    print(" ")
    print(" Actual score: ")
    print(y_actual, "(", y_range[y_actual],")")
    print(" ")
    print("Predicted probabilities: ")
    for i in range(0,4):
        print(y_range[i], " : ", temp_pred_1[i])
    return

dt_probs(805)
dt_probs(50)

## 틀린 데이터에 대한 인덱스 출력

for i in range(len(y_test)):
    if y_pred_dt[i] != y_test2[i]:
        print(i)

X_test.head()
y_test3 = y_test.to_frame()
y_test3 = y_test3.reset_index()
y_test3.head()
y_test3.ix[19]
X_test.ix[1090]
dt_probs(1090)

y_test3.ix[41]
X_test.ix[1130]
dt_probs(1130)




####################### 예제 ################################





df = pd.read_table("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", sep=',',
                  names = ["sepal length in cm", "sepal width in cm", "petal length in cm", "petal width in cm", "class"])

df.shape
df[pd.isnull(df).any(axis=1)]
df['class'].value_counts()
car_counts = pd.DataFrame(df['class'].value_counts())
car_counts['Percentage'] = car_counts['class']/car_counts.sum()[0]


plt.figure(figsize=(8,8))
plt.pie(car_counts["Percentage"],
       labels = ['Iris-virginica','Iris-versicolor', 'Iris-setosa'])
df.head()

le = preprocessing.LabelEncoder()
encoded_class = le.fit(df['class'])
encoded_class.classes_

encoded_class.transform(['Iris-virginica']) ## 확인용 !!
encoded_class.transform(['Iris-versicolor'])
encoded_class.transform(['Iris-setosa'])
encoded_class.inverse_transform(1)

for i in range (0,3):
    print(i, ":", encoded_class.inverse_transform(i))

## 바꾸고싶은 대상을 바꾸기 문자를 숫자로 숫자를 문자로

df['e.class'] = df['class'].map(lambda x: encoded_class.transform([x]))
df.head()
df['e.class'] = df['e.class'].map(lambda x: x[0])
df.head()

##


## 함수로 만들어서 쓰기

##
    
    
pd.DataFrame(df['class'].value_counts())


X = df[['sepal length in cm', 'sepal width in cm', 'petal length in cm', 'petal width in cm']]
type(X)
X.shape

y = df['e.class']
type(y)
y.shape
len(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)
    
X_train.shape
y_train.shape
X_test.shape
y_test.shape

#Decision Trees를 이용

clf_dt = tree.DecisionTreeClassifier(random_state=10) #####################################
clf_dt.fit(X_train, y_train)                          ### 분석하는 명령은 단 3줄밖에 안된다.!!
y_pred_dt = clf_dt.predict(X_test)                    ###### y_pred_dt = 예측한 결과########
type(y_pred_dt)
y_pred_dt.shape
y_pred_dt


print(metrics.accuracy_score(y_test, y_pred_dt))
np.sum(y_test==y_pred_dt)/len(y_test)

correct_pred_dt = []
wrong_pred_dt = []

y_test2 = y_test.reset_index(drop = True)
y_test2 = y_test2.as_matrix()
for i in range(len(y_test2)):
    if y_test2[i] != y_pred_dt[i]:
        wrong_pred_dt.append(i)
    else:
        correct_pred_dt.append(i)
  
    
print("Correctly indetified labels: ", len(correct_pred_dt))
print(" ")
print("Wrong indetified labels: ", len(wrong_pred_dt)) ## 참고로, 프린트 함수는 프로그램 실행 시간이 길 때, 얼마나 완료되었는지 확인용으로도 사용 가능하다

X_test.head()
y_test[0:5]
y_pred_dt[0:5]

X_test.index
X_test.ix[82]
index_num = 82
def dt_probs(index_num):
    X_param = X_test.ix[index_num]
    X_param = X_param.to_frame()
    X_param = X_param.transpose()
    temp_pred = clf_dt.predict_proba(X_param) ## 예측에 대한 확률 [ 0 0 1 0 ] -> 언어셉터블 어셉터블 굿 베리굿 순서로 확률
    temp_pred_1 = temp_pred[0]
    y_actual = y_test[index_num]
    y_range = ['Iris-virginica','Iris-versicolor','Iris-setosa']
    print("For index number: ", index_num)
    print(" ")
    print("Fetures entered: ")
    print(X_param)
    print(" ")
    print(" Actual score: ")
    print(y_actual, "(", y_range[y_actual],")")
    print(" ")
    print("Predicted probabilities: ")
    for i in range(0,3):
        print(y_range[i], " : ", temp_pred_1[i])
    return

dt_probs(82)

for i in X_test.index:
    dt_probs(i)
    
## 틀린 데이터에 대한 인덱스 출력

for i in range(len(y_test)):
    if y_pred_dt[i] != y_test2[i]:
        print(i)

X_test.head()
y_test3 = y_test.to_frame()
y_test3 = y_test3.reset_index()
y_test3.head()
y_test3.ix[23]
X_test.ix[23]
dt_probs(134)

y_test3.ix[41]
X_test.ix[1130]
dt_probs(1130)


correct_pred_dt
wrong_pred_dt



###################### 풀이 ###############################



df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
            ,delimiter = ',',header=-1)

df.columns = ['SL','SW','PL','PW','class']



##########################################################


import graphviz
dot_data = tree.export_graphviz(clf_dt, out_file = 'C:/tree_test.dot',
            feature_names = X.columns,
            class_names = encoded_class.classes_)




import sklearn
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from IPython.display import Image
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


df = pd.read_table("https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data", sep=',',
       names = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "eval"])

le = preprocessing.LabelEncoder()

def encode_col(col_name):
    encodes = le.fit(df[col_name])
    new_col_name = "e."+col_name
    df[new_col_name] = df[col_name].map(lambda x: encodes.transform([x]))
    df[new_col_name] = df[new_col_name].map(lambda x: x[0])
    return 


encode_col('buying')
encode_col('maint')
encode_col('doors')
encode_col('persons')
encode_col('lug_boot')
encode_col('safety')
encode_col('eval')

X = df[['e.buying', 'e.maint', 'e.doors', 'e.persons', 'e.lug_boot', 'e.safety']]
y = df['e.eval']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)


clf_knn = KNeighborsClassifier(n_neighbors=5)
clf_knn.fit(X_train, y_train)
y_pred = clf_knn.predict(X_test)

print(metrics.accuracy_score(y_test,y_pred))


correct_pred = []
wrong_pred = []
y_test2 = y_test.reset_index(drop = True)
y_test2 = y_test2.as_matrix()
for i in range(0,432):
    if y_test2[i] != y_pred[i]:
        wrong_pred.append(i)
    else:
        correct_pred.append(i)
        
print("Correctly indetified labels: ", len(correct_pred))
print(" ")
print("Wrong indetified labels: ", len(wrong_pred))


X_test.head()
y_test[0:5]
y_pred[0:5]

y_test3 = y_test.to_frame()
y_test3 = y_test3.reset_index()
y_test4 = y_test3.drop('e.eval', 1)

wrong_list = [ ]

for i in wrong_pred:
    wrong_index = y_test4.iloc[i]
    wrong_index1 = wrong_index[0]
    wrong_list.append(wrong_index1)


def knn_probs(index_num):
    X_param = X_test.ix[index_num]
    X_param = X_param.to_frame()
    X_param = X_param.transpose()
    temp_pred = clf_knn.predict_proba(X_param)
    temp_pred_1 = temp_pred[0]
    y_actual = y_test[index_num]
    y_range = ['Unacceptable','Acceptable', 'Good', 'Very Good']
    print("For index number: ", index_num)
    print(" ")
    print("Fetures entered: ")
    print(X_param)
    print(" ")
    print(" Actual score: ")
    print(y_actual, "(", y_range[y_actual],")")
    print(" ")
    print("Predicted probabilities: ")
    for i in range(0,4):
        print(y_range[i], " : ", temp_pred_1[i])
    print(" ")
    if index_num in wrong_list:
        print("Label predicted: wrongly")
    else:
        print("Label predicted: Correctly")
    return


y_test[0:3]

knn_probs(805)

knn_probs(50)

knn_probs(1171)


#############


from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(max_depth=2, random_state=0)

clf.fit(X,y)
""" 이 두가지 옵션만 쓰면 된다.!
(n_estimators=10, n_jobs=1)  에스티메이터는 의사결정나무 몇개를 쓸것인지 잡스는 몇개의 CPU코어를 한번에 쓸건지

랜덤포레스트는 간단하긴 하지만, 쓸 경우, 에스티메이터가 많을 때가 많으므로 의사결정나무의 시각화에 어려움이 있다.->블랙박스 현상
"""