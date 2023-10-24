import numpy as np
import copy
import math


def zscore_normalize_features(X):
    """
    computes X, zcore normalized by column
    Args:
    X (ndarray (m,n)) : input data, m examples, n features
    Returns:
    X_norm (ndarray (m,n)): input normalized by column
    1mu (ndarray (n,)) : mean of each feature
    1sigma (ndarray (n,)) : standard deviation of each feature
    """
    mu = np.mean(X)
    sigma = np.std(X)
    X_Norm = (X-mu)/sigma
    return (X_Norm, mu, sigma)


def compute_cost(X, y, w, b):
    """
    compute cost
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
    Returns
      cost (scalar)    : cost
    """
    m = len(X)
    cost = 0.0
    
    for i in range(m):
           cost += ((w * X[i] + b) - y[i]) ** 2
        
    cost = (1 / (2 * m)) * cost
    
    return cost


def compute_gradient(X, y, w, b):
    """
    Computes the gradient for linear regression 
    Args:
      X : (ndarray Shape (m,n)) matrix of examples 
      y : (ndarray Shape (m,))  target value of each example
      w : (ndarray Shape (n,))  parameters of the model      
      b : (scalar)              parameter of the model      
    Returns
      dj_db : (scalar)             The gradient of the cost w.r.t. the parameter b. 
      dj_dw : (ndarray Shape (n,)) The gradient of the cost w.r.t. the parameters w. 
    """
    m = len(X)
    dj_db = (1 / m) * np.sum(w * X + b - y)
    dj_dw = (1 / m) * np.sum((w * X + b - y) * X, axis=0)
    
    return dj_db, dj_dw

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    w = w_in
    b = b_in
    J_history = []

    for i in range(num_iters):
        cost = cost_function(X, y, w, b)
        dj_db, dj_dw = gradient_function(X, y, w, b)
        
        w -= alpha * dj_dw
        b -= alpha * dj_db
        

    return w, b, J_history
