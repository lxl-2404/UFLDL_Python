"""
Created on 2023/2/14
UFLDL excecise 1c related softmax regression
"""
import numpy as np
from scipy import optimize as op
import matplotlib.pyplot as plt
# from mnist import MNIST
import idx2numpy as idx2np
# import cv2

def load12data():
    '''
    load training data from mnist dataset and labels have 1 added to them. Add a intepret column to the images data.
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

    tra_ima1 = np.reshape(tra_ima, (tra_ima.shape[0], tra_ima.shape[1]*tra_ima.shape[2])) #Reshape the array from 3 dimensions to 2 dimensions.
    tra_ima1 = np.c_[np.ones(tra_ima1.shape[0]),tra_ima1] #Add a column of ones as the intercept feature.

    return tra_ima1, tra_lab

def load12Tdata():
    '''
    Load test data from mnist dataset, and then filter the 1&2 images and labels.
    return: Test data in ndarray format. 
    '''
    file1 = "common/data/common/t10k-images-idx3-ubyte"
    file2 = "common/data/common/t10k-labels-idx1-ubyte"
    test_ima = idx2np.convert_from_file(file1)
    test_lab = idx2np.convert_from_file(file2)

    test_ima12 = np.reshape(test_ima, (test_ima.shape[0], test_ima.shape[1]*test_ima.shape[2]))
    test_ima12 = np.c_[np.ones(test_ima12.shape[0]),test_ima12]
    # test_lab12 = test_lab + 1   # different from MATLAB, python vector's index begins from 0.
    return test_ima12, test_lab

def sigmoid(x):
    # y = np.where(x>0, x, 0.000001)
    z = 1/(1+ np.exp(-x))
    return z

def costFunction(theta,X,Y):
    '''
    Cost Function.
    '''
    n = X.shape[1]
    m = X.shape[0]
    theta = np.reshape(X, (n,-1))
    A = [np.exp(X@theta), np.ones(m)]
    B = np.sum(A, axis=1)                    #sum elements of every row, output a array.
    B = B.reshape(-1, 1)   #reshape 之前，B的shape是(m, ),reshape之后shape是(m, 1), 不reshape的话，下行的A / B会报错
    C = np.log(A / B)
    D = sub2ind(C, Y)
    J = -np.sum(D)
    return J

def sub2ind(A, B):
    C = np.zeros(B.shape[0])
    for i in range(B.shape[0]):
        C[i] = A[i][B[i]]
    return C



def gradient(theta,X,Y):
    '''
    Gradient function.
    '''
    # m = Y.shape[0]
    grad=np.dot(X.T,sigmoid(X @ theta)-Y)
    return grad
