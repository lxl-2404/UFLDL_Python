"""
Created on 2023/3/5
UFLDL excecise related multilayer supervised neural networks
Modified:  /  /2023
"""
import numpy as np
import idx2numpy as idx2np

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

def InitializeWeights():
    '''
    Initialize Weights of multilayer neural network
    '''
    