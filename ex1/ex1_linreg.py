
#import pandas as pd
import numpy as np
from scipy import optimize as op
import matplotlib.pyplot as plt

#data = pd.read_csv("ex1/housing.data",header=None, sep='s+',engine='python')
data = np.loadtxt("ex1/housing.data")
#data = data.T
#array1 = np.ones(data.shape[0])
#print(array1.T)
data = np.c_[np.ones(data.shape[0]),data]  #add one column of ones to the data as an additional intercept feature.
#print(data[-1])
np.random.shuffle(data) #随机打乱数组顺序，如果是多维数组，numpy.random.shuffle会打乱矩阵的第一维
# print(data[-1])
Train_X = data[:400, :-1] #
Train_y = data[ :400, -1] #
Test_X = data[400: , :-1]
Test_y = data[400: , -1]
#print(Test_y)
#print(Test_y.shape)
#print(Train_y.shape)
#print(data.shape[0])
m = Train_X.shape[0]
n = Train_X.shape[1]
theta = np.random.rand(n)
print(theta.shape)
def costFunction(Theta, X, y):
    return np.square(np.linalg.norm(np.dot(X, Theta)-y))/2

def gradientFun(Theta, X, y):
    return np.dot(X.T,(np.dot(X, Theta)-y))
result = op.minimize(fun=costFunction,x0=theta,args=(Train_X, Train_y),jac=gradientFun)
print(result.x)
predicted_prices =np.dot(Test_X, result.x)

indx = np.argsort(predicted_prices) #得到升序排序后的索引序列
plt.plot(predicted_prices[indx],'r+')    #利用索引序列排序预测价格与真实价格
plt.plot(Test_y[indx],'b.',label='True')
plt.show()