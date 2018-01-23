# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""



## 피보나치
'''
n = 13
a = 0
b = 1
print('0',end="")
while b <= n:
    print(',%d'%b,end="")
    a,b = b,a+b
    
print('사과 %d 개'%b)
b = 'abcd'
print('사과 %d ro'%b)
'''
## 1부터 1000에서 각 숫자의 개수 구하기
'''
a = 1
b = 0

while a <= 1000:
    if a/10 < 1:
        b=b+1
        if a/100 < 10:
            b=b+1
            if a/1000 < 100:
                b=b+1
                print(',0 : %d개'%b)
    b = 0
    
    if a/10 == 1:
        b=b+1
        if 9 < a/100 < 20:
            b=b+1
            if 99 < a/1000 < 200:
                b=b+1
                if a==1000:
                    b=1
                    print(',1 : %d개'%b)
    b = 0
    a=a+1
    '''
'''
temp = [0,0,0,0,0,0,0,0,0,0]
for num in range(1,1001):
    for tp in str(num):
        temp[int(tp)]+=1
        
num = 100
for tp in str(num):
    print(tp)
'''
'''
temp = [0,0,0,0,0,0,0,0,0,0]
for num in range(1,1001):
    for tp in str(num):
        temp[int(tp)]+=1
'''


def fib():
    n = int(input('숫자입력 : '))

    

a = [1,2,3]
b = [4,5,6]
list(zip(a,b))
a = [1,2]
list(zip(a,b))

## 순서쌍

s = [1,3,4,8,13,17,20]
pairs = list(zip(s,s[1:]))


def plus(a,b):
    return(a+b)

lambda x : x + 2


## intersection sort

def swp(arr,i,j):
    tmp = arr[i]
    
    for num in range(i,j,-1):
        arr[num] = arr[num-1]
        
    arr[j] = tmp
    return(arr)
    
arr = [5,2,4,6,1,3]
    
for i in range(1,len(arr)):
    target = arr[i]
    pos = i
    for j in range(i-1,-1,-1):
        if target > arr[j]:
            break
        else:
            pos = j
            
        
import numpy as np

a = [1,2,3]

b= np.array(a)

a + 1 ## <<에러
b + 1

## 배열 차원 확인
b
b.ndim    
b.shape   <<- 추천
len(b)    <<- 추천

##
a=[[1,2,3],[4,5,6]]

b= np.array(a)

b.ndim
b.shape
len(b)

a1 = [[1,2,3,4,]]
a2 = [[1],[2],[3],[4]]
b1 = np.array(a1)
b2 = np.array(a2)

## 전치
b.T
c = [1,2,3]
d = np.array(c)
d.T << 1차원인 d는 전치x

e=np.eye(4)


a = np.arange(20)
a.reshape(4,-1)
a.reshape(3,-1)

np.vstack(a,a.reshape(2,-1))

np.vstack(a.reshape(4,-1),
          a.reshape(2,-1))

np.vstack(a.reshape(2,-1),
          a.reshape(2,-1))

a = [[1,2,3],[4,5,6]]
b = np.array(a)
b*b

b.dot(b)
b.dot(b.T)

redwine = np.loadtxt


import pandas as pd

a = pd.Series([1,2,3,4])
a
a+1
a*2
a/2
a.values

type(a)
type(a.values)

a = pd.Series([1,2,3,4])
b = pd.Series([1,2,3,4],
             index=['a','b','c','d'])
a
b

b = pd.Series([1,2,3,4],
             index=['a','b','c'])

a+b

b = pd.Series([1,2,3,4])

a+b

b['a']
b[0]
b['a':'c']
b[['a','c']]
b['c':'a']

## 숫자 indexing

b[b>2]
b[b==2]
b[b!=2]

sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}

a = pd.Series(sdata)

states = ['California', 'Ohio', 'Oregon', 'Texas']

b = pd.Series(sdata, index=states)

pd.isnull(a)
pd.notnull(a)

a+b

a.name = 'population'
a.index.name = 'state'

data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
           'year': [2000, 2001, 2002, 2001, 2002],
           'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}

pd.DataFrame(data)

df = pd.DataFrame(data)
df2 = pd.DataFrame(data, 
                   columns=['year', 'state', 'pop', 'debt'],
                   index=['one', 'two', 'three', 'four', 'five'])
df2

df2['debt'] = 1
df2.debt = 11
df2.debt = np.arange(5)
df2.debt = [10,5,4,2,14]

redwine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv', delimiter=';')

redwine.min( )
redwine.mean( )
redwine.std( ) 

pd.read_sql

redwine.corr()
redwine.sum

redwine.corr( )
redwine[['alcohol','quality','pH']].corr( )

## group by
redwine_group = redwine['alcohol'].groupby(redwine['quality'])
redwine_group.mean( )
redwine_group.std( )

## matplot

import matplotlib as mpl
import matplotlib.pylab as plt

a = [1,4,9,16]
b = np.array(a)
c = pd.Series(a)
plt.plot(c)

plt.plot([10,20,30,40],[1,4,9,2])

plt.plot([1,4,9,16], 'rs--') ## 빨간 점선

x = np.linspace(-np.pi, np.pi, 256)
y = np.cos(x)
plt.plot(x, y)
plt.xticks([-np.pi, -np.pi/2, 0, np.pi / 2, np.pi])
plt.yticks([-1, 0, +1])

x = np.linspace(-np.pi, np.pi, 256)
y = np.cos(x)
plt.plot(x, y)
plt.xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
           [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])
plt.yticks([-1, 0, 1], ["Low", "Zero", "High"])
plt.grid(True)


x = np.linspace(-np.pi, np.pi, 256)
c, s = np.cos(x), np.sin(x)
plt.plot(x, c, ls="--", label="cosine")
plt.plot(x, s, ls=":", label="sine")
plt.legend(loc=5)

x = np.linspace(-np.pi, np.pi, 10)
c, s = np.cos(x), np.sin(x)
plt.plot(x, c, ls="--", label="cosine")
plt.plot(x, s, ls=":", label="sine")
plt.legend(loc=5)

x1 = [1,2,3]
x2 = [2,1,2]
x3 = [3,2,1]
x4 = [1,2,1]

ax1 = plt.subplot(2,2,1)
plt.plot(x1)
print(ax1)
ax2 = plt.subplot(2,2,2)
plt.plot(x2)
print(ax2)
ax3 = plt.subplot(2,2,3)
plt.plot(x3)
print(ax3)
ax4 = plt.subplot(2,2,4)
plt.plot(x4)
print(ax4)


y = [2,3,1]
x = np.arange(len(y))
xlabel = ['가', '나', '다']
plt.bar(x,y)
plt.xticks(x, xlabel)

plt.barh(x,y)
plt.yticks(x, xlabel)
plt.show()

men = [20, 35, 30, 35, 27]
women = [25, 32, 34, 20, 25]
x = np.arange(len(men))
plt.bar(x,men,width=0.3,label='men')
plt.bar(x+0.3,women,width=0.3,label='women')
plt.xticks(x+0.15, np.arange(len(men)))
plt.legend(loc=0)


data = np.random.normal(0,6,100)
plt.boxplot(data)

mu = 100
sigma = 15
x = mu+sigma*np.random.randn(1000)
plt.hist(x,30)

plt.hist(x,10000)

x =np.arange(-50,50)
y1 = x + np.random.randn(100)*1000
plt.scatter(x,y1)
y2 = x**2+np.random.randn(100)*1000
plt.scatter(x,y2)

df = pd.DataFrame([x,y1,y2]).T
df.corr()







