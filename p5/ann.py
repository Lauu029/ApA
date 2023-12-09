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


"""Parámetros
    theta1 : array_like
        Pesos para la primera capa en la red neuronal.
        Tiene forma (tamaño de la 2ª capa oculta x tamaño de la entrada)

    theta2: array_like
        Pesos para la segunda capa en la red neuronal.
        Tiene forma (tamaño de la capa de salida x tamaño de la 2ª capa oculta)

    X : array_like
        Las entradas de la imagen tienen forma (número de ejemplos x dimensiones de la imagen).

    Retorno
        p : array_like
        Vector de predicciones que contiene la etiqueta predicha para cada ejemplo.
        Tiene una longitud igual al número de ejemplos.
"""
#Predecir la etiqueta de una entrada dada una red neuronal entrenada.
def predict(theta1, theta2, X):
    a1,a2,a3=FeedForward(theta1,theta2,X)
    p = np.argmax(a3, axis=1)
    return p

#Cálculo del coste sumando la regularización L2
def reg_cost(theta1, theta2, X, y, lambda_):
    m = len(X)
    c = cost(theta1,theta2,X,y)
    regularizacion = (lambda_ / (2 * m)) * (np.sum(theta1[:, 1:]**2) + np.sum(theta2[:, 1:]**2))
    
    J = c + regularizacion
    return J


"""Parámetros
   theta1 : array_like
        Pesos para la primera capa en la red neuronal.
        Tiene forma (tamaño de la 2ª capa oculta x tamaño de la entrada + 1)

    theta2: array_like
        Pesos para la segunda capa en la red neuronal.
        Tiene forma (tamaño de la capa de salida x tamaño de la 2ª capa oculta + 1)

    X : array_like
        Las entradas tienen forma (número de ejemplos x número de dimensiones).

    y : array_like
    Codificación 1-hot de las etiquetas para la entrada, con forma
    (número de ejemplos x número de etiquetas).

    lambda_ : float
        El parámetro de regularización.

    Retorno
        J : float
        El valor calculado para la función de costo.
"""
#Coste de la red de neuronas de dos capas sin regularización
def cost(theta1,theta2 , X, y):
    z2,a1,a2,a3 = FeedForward(theta1, theta2, X)
    m = len(X)
    term1 = np.sum(y * np.log(a3))
    term2 = np.sum((1 - y) * np.log(1 - a3))

    J = (-1 / m) * (term1 + term2)
    return J



"""  Toma la salida de la red neuronal (h), las etiquetas reales (y), y la derivada de la función de activación (funDer).
Devuelve un vector de deltas.
"""
# Calcula los deltas para la capa de salida.
def DeltaLast(h,y,funDer):
    D = np.zeros(h.shape[0])
    for j in range(h.shape[0]):
        D[j] = (h[j]-y[j])*funDer(h[j])
    return D

"""Toma los pesos de la capa actual (thetaL), los deltas de la capa siguiente (deltaNext), las activaciones de la capa actual (aL), y la derivada de la función de activación (funDer). Devuelve un vector de deltas."""
#Calcula los deltas para las capas ocultas.
def Delta(thetaL,deltaNext,aL,funDer):
    D = np.zeros(aL.shape[0])
    for i in range(aL.shape[0]):
        for j in range(deltaNext.shape[0]):
            D[i] += thetaL[j,i]*deltaNext[j]*funDer(aL[i])
    return D

"""La función utiliza bucles para acumular los productos de los deltas y las activaciones en las matrices de gradientes gradientTetha1 y gradientTetha2.

Devuelve las matrices de gradientes actualizadas,que se usan en  el proceso de actualización de los pesos durante el entrenamiento de la red"""
# Acumula los gradientes para los pesos de la red neuronal
def Gradientes(gradientTetha1,gradientTetha2,delta2,delta3,a1,a2):
    for j in range(delta3.shape[0]):
        for k in range(a2.shape[0]):
            gradientTetha2[j,k] += delta3[j]*a2[k]
    for j in range(delta2.shape[0]):
        for k in range(a1.shape[0]):
            gradientTetha1[j,k] += delta2[j]*a1[k]
    return gradientTetha1, gradientTetha2
"""Parámetros
    theta1 : array_like
        Pesos para la primera capa en la red neuronal.
        Tiene forma (tamaño de la 2ª capa oculta x tamaño de la entrada + 1)

    theta2: array_like
        Pesos para la segunda capa en la red neuronal.
        Tiene forma (tamaño de la capa de salida x tamaño de la 2ª capa oculta + 1)

    X : array_like
        Las entradas tienen forma (número de ejemplos x número de dimensiones).

    y : array_like
        Codificación 1-hot de las etiquetas para la entrada, con forma
        (número de ejemplos x número de etiquetas).

    lambda_ : float
        El parámetro de regularización.

    Retorno
    J : float
        El valor calculado para la función de costo.

    grad1 : array_like
    Gradiente de la función de costo con respecto a los pesos para la primera capa en la red neuronal, theta1.
    Tiene forma (tamaño de la 2ª capa oculta x tamaño de la entrada + 1)

    grad2 : array_like
        Gradiente de la función de costo con respecto a los pesos para la segunda capa en la red neuronal, theta2.
        Tiene forma (tamaño de la capa de salida x tamaño de la 2ª capa oculta + 1)
"""
#Calcular el coste y el gradiente para una red neuronal de 2 capas.
def backprop(theta1, theta2, X, y, lambda_):
    m = len(X)
    z2,a1,a2,a3= FeedForward(theta1, theta2, X)
    
    cost = reg_cost(theta1, theta2, X, y, lambda_)
    # Retropropagación
    delta3 = a3 - y
   
    delta2 = np.dot(delta3, theta2[:, 1:]) * sigmoid_der(z2)

    # Cálculo de gradientes
    grad1 = (1 / m) * np.dot(delta2.T, a1)
    grad2 = (1 / m) * np.dot(delta3.T, a2)
    
    # Regularización de gradientes
    grad1[:, 1:] += (lambda_ / m) * theta1[:, 1:]
    grad2[:, 1:] += (lambda_ / m) * theta2[:, 1:]
    return cost, grad1, grad2
    
    
