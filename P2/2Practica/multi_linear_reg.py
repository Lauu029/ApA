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
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)

    X_norm = (X-mu)/sigma
    
    return (X_norm, mu, sigma)

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
    """m = len(X)
    total_cost = 0.0
    
    for i in range(m):
        total_cost += ((w * X[i] + b) - y[i]) ** 2
        
    total_cost = (1 / (2 * m)) * total_cost"""
    m = len(X)
    total_cost = 0.0
    
    for i in range(m):
        temp =0
        for j in range (len(w)):
            temp+=w[j]*X[i][j]
        
        temp+=b
        total_cost += (temp - y[i]) ** 2
        
    total_cost = (1 / (2 * m)) * total_cost
    
    return total_cost


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
    m = len(y)
    dj_dw_temp = (1/m) * np.dot(X.T, (np.dot(X, w) + b)-y)
    dj_db = (1/m) * np.sum( (np.dot(X, w)+b)-y)
  
    dj_dw = dj_dw_temp.flatten()
    return dj_db, dj_dw

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    """
    Performs batch gradient descent to learn theta. Updates theta by taking
    num_iters gradient steps with learning rate alpha
    Args:
    X : (array_like Shape (m,n) matrix of examples
    y : (array_like Shape (m,)) target value of each example
    w_in : (array_like Shape (n,)) Initial values of parameters of the
    model
    b_in : (scalar) Initial value of parameter of the model
    cost_function: function to compute cost
    2
    gradient_function: function to compute the gradient
    alpha : (float) Learning rate
    num_iters : (int) number of iterations to run gradient descent
    Returns
    w : (array_like Shape (n,)) Updated values of parameters of the model
    after running gradient descent
    b : (scalar) Updated value of parameter of the model
    after running gradient descent
    J_history : (ndarray): Shape (num_iters,) J at each iteration,
    primarily for graphing later
    """
    # Inicializar los parámetros w y b con los valores dados
    w = w_in
    b = b_in
    
    # Crear un arreglo vacío para almacenar el historial de costo
    J_history = np.zeros(num_iters)
    
    # Obtener el número de ejemplos m y el número de variables n de la matriz X
    m, n = X.shape
    
    # Repetir el proceso num_iters veces
    for i in range(num_iters):
        
        # Calcular el costo y el gradiente usando las funciones dadas
        cost = cost_function(X, y, w, b)
        dj_db, dj_dw = gradient_function(X, y, w, b)
        
        # Actualizar los parámetros w y b usando la regla de actualización
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        
        # Guardar el costo en el historial de costo
        J_history[i] = cost
      

    return w, b, J_history