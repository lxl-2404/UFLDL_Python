
#import pandas as pd
import numpy as np

#data = pd.read_csv("ex1/housing.data",header=None, sep='s+',engine='python')
data = np.loadtxt("ex1/housing.data")
#data = data.T
#array1 = np.ones(data.shape[0])
#print(array1.T)
data = np.c_[np.ones(data.shape[0]),data]  #add one column of ones to the data as an additional intercept feature.
print(data[-1])
np.random.shuffle(data) #随机打乱数组顺序，如果是多维数组，打乱第一维
print(data[-1])
Train_X = data[:400, :-2]
Train_y = data[ :400, -1]
Test_X = data[400: , :-2]
Test_y = data[400: , -1]
print(Test_y)
print(Test_y.shape)
print(Train_y.shape)
print(data.shape[0])
m = Train_X.shape[0]
n = Train_X.shape[1]
