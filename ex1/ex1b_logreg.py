"""
Created on 2023/2/3
UFLDL excecise 1a related linear regression
"""
import numpy as np
from scipy import optimize as op
import matplotlib.pyplot as plt
# from mnist import MNIST
import idx2numpy as idx2np
import cv2

def load12data():
    '''
    load training data from mnist dataset and then filter the 1 and 2 images and labels.
    return: training data in ndarray format.
    '''
    file1 = "common/data/common/train-images-idx3-ubyte"
    file2 = "common/data/common/train-labels-idx1-ubyte"
    tra_ima = idx2np.convert_from_file(file1)             #Use the idx2numpy package to convert the dataset
    tra_lab = idx2np.convert_from_file(file2)
    # cv2.imshow("Image", tra_ima[9])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # # x = data[:,0]
    # print(tra_lab[9])
    tempindex = np.where(tra_lab < 2 )
    tra_ima1 = tra_ima[tempindex]
    tra_ima1 = np.reshape(tra_ima1, (tra_ima1.shape[0], tra_ima1.shape[1]*tra_ima1.shape[2]))
    tra_lab1 = tra_lab[tempindex]    

    return tra_ima1, tra_lab1

def load12Tdata():
    '''
    Load test data from mnist dataset, and then filter the 1&2 images and labels.
    return: Test data in ndarray format. 
    '''
    file1 = "common/data/common/t10k-images-idx3-ubyte"
    file2 = "common/data/common/t10k-labels-idx1-ubyte"
    test_ima = idx2np.convert_from_file(file1)
    test_lab = idx2np.convert_from_file(file2)
    tempindex = np.where(test_lab < 2)  #Choose numbers 1 and 2 and get their index array. 
    test_lab12 = test_lab[tempindex]    #Use above index to filter images and relative labels.
    test_ima12 = test_ima[tempindex]
    test_ima12 = np.reshape(test_ima12, (test_ima12.shape[0], test_ima12.shape[1]*test_ima12.shape[2]))

    return test_ima12, test_lab12

def sigmoid(x):
    z=1/(1+np.exp(-x))
    return z

def costFunction(theta,X,Y):
    m=X.shape[0]
    J = (-np.dot(Y.T, np.log(sigmoid(X.dot(theta)))) - \
         np.dot((1 - Y).T, np.log(1 - sigmoid(X.dot(theta))))) / m
    return J


def gradient(theta,X,Y):
    m,n=X.shape
    theta=theta.reshape((n,1))
    grad=np.dot(X.T,sigmoid(X.dot(theta))-Y)/m
    return grad.flatten()

trainI12, trainL12 = load12data()

# trainI12 = np.c_[np.ones(trainI12.shape[0]),trainI12]
testI12, testL12 = load12Tdata()

print(trainI12.shape)
print(testI12.shape)
n = 
theta = np.random.rand(n)
result = op.minimize(fun=costFunction,x0=theta,args=(Train_X, Train_y),jac=gradient)