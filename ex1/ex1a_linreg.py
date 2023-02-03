"""
Created on 2023/2/1
UFLDL excecise 1a related linear regression
"""
#import pandas as pd
import numpy as np
from scipy import optimize as op
import matplotlib.pyplot as plt

#data = pd.read_csv("ex1/housing.data",header=None, sep='s+',engine='python')
data = np.loadtxt("ex1/housing.data")

data = np.c_[np.ones(data.shape[0]),data]  #add one column of ones to the data as an additional intercept feature.

np.random.shuffle(data) #随机打乱数组顺序，如果是多维数组，numpy.random.shuffle会打乱矩阵的第一维

Train_X = data[:400, :-1] #Training data is fist 400 rows.
Train_y = data[ :400, -1] #
Test_X = data[400: , :-1]  #Test data is rows after 400's.
Test_y = data[400: , -1]

m = Train_X.shape[0]     #numbers of samples
n = Train_X.shape[1]     #numbers of features
theta = np.random.rand(n)
print(theta.shape)
def costFunction(Theta, X, y):
    return np.square(np.linalg.norm(np.dot(X, Theta)-y))/2

def gradientFun(Theta, X, y):
    return np.dot(X.T,(np.dot(X, Theta)-y))
result = op.minimize(fun=costFunction,x0=theta,args=(Train_X, Train_y),jac=gradientFun)   #Call the minimize function to regression
print(result)
predicted_prices =np.dot(Test_X, result.x)

indx = np.argsort(predicted_prices) #得到升序排序后的索引序列
line1 = plt.plot(predicted_prices[indx],'r+')    #利用索引序列排序预测价格与真实价格
line2 = plt.plot(Test_y[indx],'b.')


# plt.legend((line1,line2),['Predicted','True'],loc = 0, 
#            title='legends', ncol=2, markerfirst=False,
#            numpoints=2, frameon=True, fancybox=True,
#            facecolor='gray', edgecolor='r', shadow=True)

plt.show()