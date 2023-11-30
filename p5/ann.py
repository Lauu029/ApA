import numpy as np

def DeltaLast(h,y,funDer):
    D = np.zeros(h.shape[0])
    for j in range(h.shape[0]):
        D[j] = (h[j]-y[j])*funDer(h[j])
    return D
def Delta(thetaL,deltaNext,aL,funDer):
    D = np.zeros(aL.shape[0])
    for i in range(aL.shape[0]):
        for j in range(deltaNext.shape[0]):
            D[i] += thetaL[j,i]*deltaNext[j]*funDer(aL[i])
    return D
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def FeedForward(theta1, theta2, X):
    a1= np.array(X)
    a1 = np.hstack([np.ones((a1.shape[0], 1)), a1])
    z2 = np.dot(a1, theta1.T)
    a2 = sigmoid(z2)
    a2 = np.hstack([np.ones((a2.shape[0], 1)), a2])
    z3 = np.dot(a2, theta2.T)
    a3 = sigmoid(z3)
    return a1, a2, a3

def predict(theta1, theta2, X):
    """
    Predict the label of an input given a trained neural network.

    Parameters
    ----------
    theta1 : array_like
        Weights for the first layer in the neural network.
        It has shape (2nd hidden layer size x input size)

    theta2: array_like
        Weights for the second layer in the neural network. 
        It has shape (output layer size x 2nd hidden layer size)

    X : array_like
        The image inputs having shape (number of examples x image dimensions).

    Return 
    ------
    p : array_like
        Predictions vector containing the predicted label for each example.
        It has a length equal to the number of examples.
    """
    a1,a2,a3=FeedForward(theta1,theta2,X)
    p = np.argmax(a3, axis=1)
    return p
def reg_cost(theta1, theta2, X, y, lambda_):
    m = len(X)
    c = cost(theta1,theta2,X,y,lambda_)
    regularizacion = (lambda_ / (2 * m)) * (np.sum(theta1[:, 1:]**2) + np.sum(theta2[:, 1:]**2))
    
    J = c + regularizacion
    return J

def cost(theta_list , X, y, lambda_):
    """
    Compute cost for 2-layer neural network. 

    Parameters
    ----------
    theta1 : array_like
        Weights for the first layer in the neural network.
        It has shape (2nd hidden layer size x input size + 1)

    theta2: array_like
        Weights for the second layer in the neural network. 
        It has shape (output layer size x 2nd hidden layer size + 1)

    X : array_like
        The inputs having shape (number of examples x number of dimensions).

    y : array_like
        1-hot encoding of labels for the input, having shape 
        (number of examples x number of labels).

    lambda_ : float
        The regularization parameter. 

    Returns
    -------
    J : float
        The computed value for the cost function. 

    """
    a1,a2,a3 = FeedForward(theta1, theta2, X)
    m = len(X)
    term1 = np.sum(y * np.log(a3))
    term2 = np.sum((1 - y) * np.log(1 - a3))
    regularizacion = (lambda_ / (2 * m)) * (np.sum(theta1[:, 1:]**2) + np.sum(theta2[:, 1:]**2))

    J = (-1 / m) * (term1 + term2)#+ regularizacion
    return J

def Gradientes(gradientTetha1,gradientTetha2,delta2,delta3,a1,a2):
    for j in range(delta3.shape[0]):
        for k in range(a2.shape[0]):
            gradientTetha2[j,k] += delta3[j]*a2[k]
    for j in range(delta2.shape[0]):
        for k in range(a1.shape[0]):
            gradientTetha1[j,k] += delta2[j]*a1[k]
    return gradientTetha1, gradientTetha2

def backprop(theta1, theta2, X, y, lambda_):
    """
    Compute cost and gradient for 2-layer neural network. 

    Parameters
    ----------
    theta1 : array_like
        Weights for the first layer in the neural network.
        It has shape (2nd hidden layer size x input size + 1)

    theta2: array_like
        Weights for the second layer in the neural network. 
        It has shape (output layer size x 2nd hidden layer size + 1)

    X : array_like
        The inputs having shape (number of examples x number of dimensions).

    y : array_like
        1-hot encoding of labels for the input, having shape 
        (number of examples x number of labels).

    lambda_ : float
        The regularization parameter. 

    Returns
    -------
    J : float
        The computed value for the cost function. 

    grad1 : array_like
        Gradient of the cost function with respect to weights
        for the first layer in the neural network, theta1.
        It has shape (2nd hidden layer size x input size + 1)

    grad2 : array_like
        Gradient of the cost function with respect to weights
        for the second layer in the neural network, theta2.
        It has shape (output layer size x 2nd hidden layer size + 1)

    """

    #generamos los deltas
    delta3 = DeltaLast(a3,y,funDer)
    delta2 = Delta(theta2,delta3,a2,funDer)
    #Generamos los gradientes
    gradientTetha1, gradientTetha2 = Gradientes(gradientTetha1,gradientTetha2,delta2,delta3,a1,a2)
    for j in range(delta3.shape[0]):
        for k in range(a2.shape[0]):
            theta2[j,k] = theta2[j,k] - alpha * gradientTetha2[j,k]/m
    for j in range(delta2.shape[0]):
        for k in range(a1.shape[0]):
            theta1[j,k] = theta1[j,k] - alpha * gradientTetha1[j,k]/m

    return (J, grad1, grad2)

