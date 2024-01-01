import numpy as np
import copy
import math


def sigmoid(z):
    """
    Compute the sigmoid of z

    Args:
        z (ndarray): A scalar, numpy array of any size.

    Returns:
        g (ndarray): sigmoid(z), with the same shape as z

    """   
    g= 1/(1+ np.exp(-z))
    return g


#########################################################################
# logistic regression
#
def compute_cost(X, y, w, b, lambda_=None):
    """
    Computes the cost over all examples
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (array_like Shape (m,)) target value
      w : (array_like Shape (n,)) Values of parameters of the model
      b : scalar Values of bias parameter of the model
      lambda_: unused placeholder
    Returns:
      total_cost: (scalar)         cost
    """
    m,n = X.shape
    
    g = sigmoid(X @ w + b)
    
    loss = (-y * np.log(g)) - (1-y)*np.log(1-g)
    
    total_cost = 1/m * np.sum(loss)
    
    return total_cost


def compute_gradient(X, y, w, b, lambda_=None):
    """
    Computes the gradient for logistic regression

    Args:
      X : (ndarray Shape (m,n)) variable such as house size
      y : (array_like Shape (m,1)) actual value
      w : (array_like Shape (n,1)) values of parameters of the model
      b : (scalar)                 value of parameter of the model
      lambda_: unused placeholder
    Returns
      dj_db: (scalar)                The gradient of the cost w.r.t. the parameter b.
      dj_dw: (array_like Shape (n,1)) The gradient of the cost w.r.t. the parameters w.
    """
    m,n = X.shape
    dj_dw = 1/m *( sigmoid(X @ w + b)-y) @ X
    
    dj_db = 1/m * np.sum(sigmoid(X @ w + b)-y)
    return dj_db, dj_dw


#########################################################################
# regularized logistic regression
#
def compute_cost_reg(X, y, w, b, lambda_=1):
    """
    Computes the cost over all examples
    Args:
      X : (array_like Shape (m,n)) data, m examples by n features
      y : (array_like Shape (m,)) target value 
      w : (array_like Shape (n,)) Values of parameters of the model      
      b : (array_like Shape (n,)) Values of bias parameter of the model
      lambda_ : (scalar, float)    Controls amount of regularization
    Returns:
      total_cost: (scalar)         cost 
    """
    cost = 0
    w_total = np.sum((w**2))

    m = len(X)

    cost = np.sum(-y * np.log(sigmoid(X @ w + b)) - (1-y) * np.log(1 - sigmoid(X @ w + b)))

    total_cost = (cost/m) + ((lambda_*w_total)/(2*m))
    return total_cost

def compute_gradient_reg(X, y, w, b, lambda_=1):
    """
    Computes the gradient for linear regression 

    Args:
      X : (ndarray Shape (m,n))   variable such as house size 
      y : (ndarray Shape (m,))    actual value 
      w : (ndarray Shape (n,))    values of parameters of the model      
      b : (scalar)                value of parameter of the model  
      lambda_ : (scalar,float)    regularization constant
    Returns
      dj_db: (scalar)             The gradient of the cost w.r.t. the parameter b. 
      dj_dw: (ndarray Shape (n,)) The gradient of the cost w.r.t. the parameters w. 

    """
    dj_db = 0
    dj_dw = np.zeros(len(X[0]))
    m = len(X) 

    dj_db = np.sum(sigmoid(X @ w + b) - y)
    dj_dw = (sigmoid(X @ w + b) - y) @ X
    
    dj_db= dj_db/m
    dj_dw= (dj_dw / m) + (np.dot(np.divide(lambda_, m), w))
    
    return dj_db, dj_dw


#########################################################################
# gradient descent
#
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_=None):
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha

    Args:
      X :    (array_like Shape (m, n)
      y :    (array_like Shape (m,))
      w_in : (array_like Shape (n,))  Initial values of parameters of the model
      b_in : (scalar)                 Initial value of parameter of the model
      cost_function:                  function to compute cost
      alpha : (float)                 Learning rate
      num_iters : (int)               number of iterations to run gradient descent
      lambda_ (scalar, float)         regularization constant

    Returns:
      w : (array_like Shape (n,)) Updated values of parameters of the model after
          running gradient descent
      b : (scalar)                Updated value of parameter of the model after
          running gradient descent
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


#########################################################################
# predict
#
def predict(X, w, b):
    """
    Predict whether the label is 0 or 1 using learned logistic
    regression parameters w and b

    Args:
    X : (ndarray Shape (m, n))
    w : (array_like Shape (n,))      Parameters of the model
    b : (scalar, float)              Parameter of the model

    Returns:
    p: (ndarray (m,1))
        The predictions for X using a threshold at 0.5
    """
    m=len(X)
    p = np.zeros(m)

    for i in range(m) :
        if(sigmoid(np.dot(w, X[i]) + b) > 0.5) :
            p[i] = 1.0

    return p
