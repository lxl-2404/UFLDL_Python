"""
Created on 2023/2/14
UFLDL excecise 1c related softmax regression
"""
import numpy as np
from scipy import optimize as op
import matplotlib.pyplot as plt
# from mnist import MNIST
import idx2numpy as idx2np
from scipy.sparse import csr_matrix
# import cv2

def loaddata():
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

    tra_ima1 = np.reshape(tra_ima, (tra_ima.shape[0], tra_ima.shape[1]*tra_ima.shape[2]))    #Reshape the array from 3 dimensions to 2 dimensions.
    tra_ima1 = np.c_[np.ones(tra_ima1.shape[0]),tra_ima1]         #Add a column of ones as the intercept feature.

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


def costFunction(theta,X,Y):
    '''
    Cost Function.
    '''
    n = X.shape[1]
    m = X.shape[0]
    theta = np.reshape(theta, (n,-1))
    A = np.c_[np.exp(X@theta), np.ones(m)]
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
    n = X.shape[1]
    m = X.shape[0]
    theta = np.reshape(theta, (n,-1))
    A = np.c_[np.exp(X@theta), np.ones(m)]
    B = np.sum(A, axis=1)                    #sum elements of every row, output a array.
    B = B.reshape(-1, 1)   #reshape 之前，B的shape是(m, ),reshape之后shape是(m, 1), 不reshape的话，下行的A / B会报错
    C = A / B
    D = csr_matrix((np.ones(m), (np.arange(m), Y))).toarray()
    grad = X.T@(D - C)
    grad = np.delete(grad, -1, axis = 1)
    grad = grad.reshape(-1, 1)
    return grad


if __name__=='__main__':
    trainI, trainL = loaddata()
    m = trainI.shape[0]
    n = trainI.shape[1]
    theta = np.random.rand((n*9))*0.0001
    result = op.minimize(fun=costFunction, x0=theta, args=(trainI, trainL), method='BFGS', jac=gradient)
    print(result.message)
    print(result.success)