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
    tra_ima = idx2np.convert_from_file(file1)
    tra_lab = idx2np.convert_from_file(file2)
    # cv2.imshow("Image", tra_ima[9])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # # x = data[:,0]
    # print(tra_lab[9])
    tempindex = np.where(tra_lab < 2 )
    tra_ima1 = tra_ima[tempindex]
    tra_lab1 = tra_lab[tempindex]

    return tra_ima1, tra_lab1

def load12Tdata():
    '''
    Load test data from mnist dataset, and then filter the 1&2 images and labels.
    return: Test data in ndarray format. 
    '''
    file1 = "common/data/common/t10k-images-idx3-ubyte"
    file2 = "common/data/common/t10k-labels-idx1-ubyte"

trainI12, trainL12 = load12data()
print(trainL12[7])