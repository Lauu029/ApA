import numpy as np

def load_data():
    data = np.loadtxt("data/houses.txt", delimiter=',')
    X = data[:,0]
    y = data[:,1]
    return X, y

def load_data_multi():
    data = np.loadtxt("data/houses.txt", delimiter=',')
    X = data[:,:-1]
    y = data[:,-1:]
    return X, y
