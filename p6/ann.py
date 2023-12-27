import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_der(z):
    return sigmoid(z) * (1 - sigmoid(z))

#Propagación hacia delante de la red
def FeedForward(theta1, theta2, X):
    a1= np.array(X)
    a1 = np.hstack([np.ones((a1.shape[0], 1)), a1])
    z2 = np.dot(a1, theta1.T)
    a2 = sigmoid(z2)
    a2 = np.hstack([np.ones((a2.shape[0], 1)), a2])
    z3 = np.dot(a2, theta2.T)
    a3 = sigmoid(z3)
    return z2,a1, a2, a3

#Predecir la etiqueta de una entrada dada una red neuronal entrenada.
def predict(pred):    
    p = np.argmax(pred, axis=1)
    return p

#Cálculo del coste sumando la regularización L2
def reg_cost(theta1, theta2, X, y, lambda_):
    m = len(X)
    c = cost(theta1,theta2,X,y)
    thetas = np.concatenate((theta1[:,1:].flatten(), theta2[:,1:].flatten()))
    
    
    J = c + L2(thetas,X,y,lambda_)
    return J

def L2(theta, X, y, lambda_):
    m=len(y)
    L2=(lambda_/(2*m))*np.sum(np.sum(np.sum(theta**2)))
    return L2

#Coste de la red de neuronas de dos capas sin regularización
def cost(theta1,theta2 , X, y):
    z2,a1,a2,a3 = FeedForward(theta1, theta2, X)
    m = len(X)
    term1 = np.sum(y * np.log(a3))
    term2 = np.sum((1 - y) * np.log(1 - a3))

    J = (-1 / m) * np.sum(term1 + term2)
    return J

#Calcular el coste y el gradiente para una red neuronal de 2 capas.
def backprop(theta1, theta2, X, y, lambda_):
    m = len(X)
    z2,a1,a2,a3= FeedForward(theta1, theta2, X)     
    
    # Retropropagación
    delta3 = a3 - y   
    delta2 = np.dot(delta3, theta2[:, 1:]) * sigmoid_der(z2)

    # Cálculo de gradientes
    grad1 = (1 / m) * np.dot(delta2.T, a1)
    grad2 = (1 / m) * np.dot(delta3.T, a2)
    
    # Regularización de gradientes
    grad1[:, 1:] += (lambda_ / m) * theta1[:, 1:]
    grad2[:, 1:] += (lambda_ / m) * theta2[:, 1:]
    
    cost = reg_cost(theta1, theta2, X, y, lambda_)
    
    return cost, grad1, grad2
    
def gradientdescent(theta1, theta2, X, y, lambda_, num_iters, alpha):

    epsilon = 0.12

    theta1 = np.random.uniform(-epsilon, epsilon, size=(theta1.shape[0], theta1.shape[1]))
    theta2 = np.random.uniform(-epsilon, epsilon, size=(theta2.shape[0], theta2.shape[1]))

    for i in range(num_iters):
        J, grad1,grad2=backprop(theta1, theta2, X, y, lambda_)
        
        theta1=theta1-alpha * grad1
        theta2=theta2-alpha * grad2

    return theta1, theta2
