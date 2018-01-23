# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 09:05:50 2018

@author: 5-310
"""

## 와인데이터의 퀄리티 분석

import numpy as np
import pandas as pd

redwine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', delimiter=';')

redwine.head() ## 데이터의 전체적인 확인

redwine.quality.value_counts() ## 퀄리티가(분석대상) 어떻게 이루어져 있는지 확인

np.unique(redwine.quality) ## 대상 어레이의 중복값을 제외한 것을 보여주는 함수

## for의 다양한 사용법

models = {'ovr': {'name': 'One versus Rest', 'iters': [1, 3]},
          'multinomial': {'name': 'Multinomial', 'iters': [1, 3, 7]}} ## key값은 네임과 이털스

for model in models:
    print(model)
    
for _ in np.arange(5):
    print(1)
    
for i in np.arange(5):
    print(i)

a = [1,2,3,4,5]
for lis in a:
    print(lis)

for model in models:
    print(models[model])


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
X = redwine[redwine.columns[0:-1]]
y = redwine['quality']
X_train, X_test, y_train, y_test = train_test_split(X,y)

## 모형 적합 (모형 만드는 과정)
lr.fit(X_train,y_train)

## 오차 확인
y_pred = lr.predict(X_test)
np.sum(y_pred == y_test) / y_test.shape[0] ## y_test.shape[0] = 로우와 컬럼 숫자를 어레이 형태로 ,, 맞은거의 갯수를 전체 갯수로 나눈 것, 즉 맞은 확률
## >> 확률이 낮아서 설정을 계속 바꿔보았다.

X.head()
X.mean()
X.std() ## std = 표준편차

## 표준화 함수
def stand(x):
    out = (x - np.mean(x))/np.std(x)
    return(out)
    
stand(X.iloc[:,0]) ## iloc는 숫자로만 가능

X.columns

## 예제
'''
1. 하나의 컬럼을 가져온다
2. 가져온 컬럼의 데이터를 표준화 한다
3. 표준화한 데이터를 저장한다
4. 1~3의 과정을 전체 컬럼에 반복한다
'''

def stand(x):
    out = (x - np.mean(x))/np.std(x)
    return(out)

for i in range(len(X.columns)):
    X.iloc[:,i] = stand(X.iloc[:,i])
    print(X.iloc[:,i])

## 강사님 풀이

def stand(x):
    out = (x - np.mean(x))/np.std(x)
    return(out)
    
df = pd.DataFrame()
for col in X.columns:
    df[col] = stand(X[col])

df['alcohol']

df['aaa'] = np.arange(len(df)) ## 존재하지 않는 데이터 프레임 이름을 입력하고 이렇게 하면 새로운 데이터 프레임이 입력된다 단, 길이가 같아야 한다


## 표준화 작업을 한 데이터로 다시해보기
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df,y)

lr = LogisticRegression(solver='newton-cg',
                        multi_class='multinomial',
                        C=1,
                        penalty='l2',
                        )

lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
np.sum(y_pred == y_test) / y_test.shape[0]


## 텐서플로우 !! << 아나콘다에 없으므로 커멘드창에 입력해서 설치해야한다 pip install tensorflow


import tensorflow as tf


sess = tf.Session()
hello = tf.constant("Hello World")
print(sess.run(hello))
sess.run(hello)

a = tf.constant(20)
b = tf.constant(22)
print('a + b = {0}'.format(sess.run(a+b)))
sess.run(a+b)

#######################################################

import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

num_house = 160
np.random.seed(42)
house_size = np.random.randint(low=1000, high=3500, size=num_house )

np.random.seed(42)
house_price = house_size * 100.0 + np.random.randint(low=20000, high=70000, size=num_house)  

plt.plot(house_size, house_price, "bx")
plt.ylabel("Price")
plt.xlabel("Size")
plt.show()

#값 표준화 (언더플로우/오버플로우 방지)
def normalize(array):
    return (array - array.mean()) / array.std()

# 훈련샘플, 0.7 = 70%. 
num_train_samples = math.floor(num_house * 0.7)

# 훈련데이터 정의
train_house_size = np.asarray(house_size[:num_train_samples])
train_price = np.asanyarray(house_price[:num_train_samples:]) ## train_test_split 안쓰는 이유 : house size, house price가 인덱스없이 그냥 리스트 형태라서

train_house_size_norm = normalize(train_house_size)
train_price_norm = normalize(train_price)

# 테스트 데이터 정의
test_house_size = np.array(house_size[num_train_samples:])
test_house_price = np.array(house_price[num_train_samples:])

test_house_size_norm = normalize(test_house_size)
test_house_price_norm = normalize(test_house_price)

'''
[Tensors Type]
Constant : 상수 값
Variable : 그래프에서 조정된 값
PlaceHolder : 데이터를 그래프로 전달하는데 사용
'''

'''
## 0으로 초기화되어있는 변수생성
# Variable로 선언할 경우 initializer에 의해서 초기화됨
state = tf.Variable(0,name='counter') ## state엔 0이들어간다

## state에 1을 더하는 작업 생성
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)
# 앞의 state를 new_value로 업데이트하는 함수

## 초기화
init = tf.global_variables_initializer()

with tf.Session() as sess:
# sess = tf.Session() 랑 동일한 표현, 단 이 구문의 경우 sess.close()를 해줘야함
    # init 연산(변수 초기화)
    sess.run(init)
    # state 초기값 출력
    prunt(sess.run(init))
    # state 갱신 연산실행 뒤 출력
    for _ in range(3):
        print(sess.run(update))
        print(sess.run(state))

sess = tf.Session()
sess.run(init)
    # state 초기값 출력
    sess.run(state)
    # state 갱신 연산실행 뒤 출력
    for _ in range(3):
        sess.run(update)
        
sess.run(state)
# 변수를 초기화함. Constant로 선언한 one은 초기화되지 않음
sess.run(init)
sess.close()

'''
# placeholder
'''
input 1 = tf.placeholder(tf.float32)
input 2 = tf.placeholder(tf.float32)
input 3 = tf.multiply(input1,input2)

with tf.Session() as sess:
    print(sess.run([output],feed_dict={
    input1:[7],input2:[2]}))
    
sess = tf.Session()
sess.run([output],feed_dict={input1:[7],input2[2]})
sess.close()
'''


tf_house_size = tf.placeholder("float", name="house_size")
tf_price = tf.placeholder("float", name="price")

# 변수 준비 및 정규분포를 기반으로 값 초기화
tf_size_factor = tf.Variable(np.random.randn(), name="size_factor")
tf_price_offset = tf.Variable(np.random.randn(), name="price_offset")

# 예측 값에 대한 연산들 정의 (예측가격 = size_factor * house_size) + price_offset
tf_price_pred = tf.add(tf.multiply(tf_size_factor, tf_house_size), tf_price_offset)

# 오차 계산 방법 정의
tf_cost = tf.reduce_sum(tf.pow(tf_price_pred-tf_price, 2))/(2*num_train_samples)

# Optimizer 학습속도
learning_rate = 0.1

# 비용 손실을 최소화하는 Gradient descent optimizer 정의
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_cost)

# 훈련
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    display_every = 2  ## 짝수번에 해당하는 시도에만, 수치들을 표시해달라 !
    num_training_iter = 50 ## 트레이닝 과정을 50번 ! (계산 반복) ->짝수만 표현하므로 25개가 나온다 !

    fit_num_plots = math.floor(num_training_iter/display_every)
    fit_size_factor = np.zeros(fit_num_plots)
    fit_price_offsets = np.zeros(fit_num_plots)
    fit_plot_idx = 0 

     # 훈련데이터 반복 실행
    for iteration in range(num_training_iter):

        # 훈련데이터 Fit
        for (x, y) in zip(train_house_size_norm, train_price_norm):
            sess.run(optimizer, feed_dict={tf_house_size: x, tf_price: y})

        # 현재상태 표시
        if (iteration + 1) % display_every == 0:
            c = sess.run(tf_cost, feed_dict={tf_house_size: train_house_size_norm, tf_price:train_price_norm})
            print("iteration #:", '%04d' % (iteration + 1), "cost=", "{:.9f}".format(c), \
                "size_factor=", sess.run(tf_size_factor), "price_offset=", sess.run(tf_price_offset))
            fit_size_factor[fit_plot_idx] = sess.run(tf_size_factor)
            fit_price_offsets[fit_plot_idx] = sess.run(tf_price_offset)
            fit_plot_idx = fit_plot_idx + 1

    print("Optimization Finished!")
    training_cost = sess.run(tf_cost, feed_dict={tf_house_size: train_house_size_norm, tf_price: train_price_norm})
    print("Trained cost=", training_cost, "size_factor=", sess.run(tf_size_factor), "price_offset=", sess.run(tf_price_offset), '\n')

    train_house_size_mean = train_house_size.mean()
    train_house_size_std = train_house_size.std()
    train_price_mean = train_price.mean()
    train_price_std = train_price.std()


    plt.rcParams["figure.figsize"] = (10,8)
    plt.figure()
    plt.ylabel("Price")
    plt.xlabel("Size (sq.ft)")
    plt.plot(train_house_size, train_price, 'go', label='Training data')
    plt.plot(test_house_size, test_house_price, 'mo', label='Testing data')
    plt.plot(train_house_size_norm * train_house_size_std + train_house_size_mean,
             (sess.run(tf_size_factor) * train_house_size_norm + sess.run(tf_price_offset)) * train_price_std + train_price_mean,
             label='Learned Regression')
 
    plt.legend(loc='upper left')
    plt.show()




for (x, y) in zip(train_house_size_norm, train_price_norm):
    print(x, y)




##
    
    
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# MNIST 사이트로부터 데이터셋을 가져옴
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)


mnist.train.labels[0]
mnist.train.images[1]
mnist.train.images[1].shape
np.sqrt(784) ## 28개 이미지?
mnist.train.images.shape

arr = np.array(mnist.train.images[1])
import matplotlib.pyplot as plt
plt.imshow(arr.reshape([28,28]))

# 28 X 28 이미지 데이터를 위한 Placeholder
x = tf.placeholder(tf.float32, shape=[None, 784])
# 각 요소의 예상확률을 포함하는 10개의 요소를 가진 벡터 (자릿수 (0-9)
y_ = tf.placeholder(tf.float32, [None, 10])

# Weight & balance 정의
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 추론
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 오차 추정
cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 1000개 훈련
for I in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# 모델이 얼마나 잘 수행했는지 평가. (실제 y, 예측 y_)
Correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
Accuracy = tf.reduce_mean(tf.cast(Correct_prediction, tf.float32))
Test_accuracy = sess.run(Accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
print('Test Accuracy: {0}%'.format(Test_accuracy * 100.0))
sess.close()



































































