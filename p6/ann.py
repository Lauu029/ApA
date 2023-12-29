import numpy as np

#No lo tienes generalizado a multiples capas y neuronas por capa, para la práctica 6 es necesario. Os recomiendo que le paseis al algoritmo un vector que indique el numero de capas ocultas y el número de neuronas por capa. Por ejemplo [ 10, 3, 4] podría indicar una capa de 10 neuronas, otra de 3 y otra de 4 (contando la entrada o no en función de como lo implementeis). Por lo demás está muy bien, enhorabuena

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_der(z):
    return sigmoid(z) * (1 - sigmoid(z))


#Propagación hacia delante de la red
def FeedForward(theta1, theta2, X):
    a1 = np.array(X)
    a1 = np.hstack([np.ones((a1.shape[0], 1)), a1])
    z2 = np.dot(a1, theta1.T)
    a2 = sigmoid(z2)
    a2 = np.hstack([np.ones((a2.shape[0], 1)), a2])
    z3 = np.dot(a2, theta2.T)
    a3 = sigmoid(z3)
    return z2,a1, a2, a3

#Propagación hacia delante de la red para m capas con n neuronas cada capa
def FeedForward_generalizado(thetas, nNeuronas_capa, X):
    m = len(nNeuronas_capa)
    z = []  # Lista para almacenar las entradas ponderadas
    a = []  # Lista para almacenar las activaciones
    #Neuronas de la capa de entrada
    a1= np.array(X)
    a1 = np.hstack([np.ones((a1.shape[0], 1)), a1])
    
    #neuronas de las capas ocultas
    for i in range (1,m-1):
        print(f"Hasta {i} ha ido bien")
        z2 = np.dot(a1, thetas[i].T)
        a2 = sigmoid(z2)
        a2 = np.hstack([np.ones((a2.shape[0], 1)), a2])
        z.append(z2)
        a.append(a2)
        a1 = a2  # Actualizar a1 para la próxima iteración
        
        
   #neuronas de las capas de salida 
    z3 = np.dot(a2, thetas[m-1].T)
    a3 = sigmoid(z3)
    z.append(z3)
    a.append(a3)
    return z, a

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
#Cálculo del coste sumando la regularización L2 para cualquier numero de capas y neuronas
def reg_cost_generalizado(thetas, nNeuronas_capa, X, y, lambda_):
    m = len(X)
    c = cost_generalizado(thetas, nNeuronas_capa,X,y)
    thetas = np.concatenate([theta[:,1:].flatten() for theta in thetas])
    
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

#Coste de la red de neuronas de m numero de capas con n numero de neuronas sin regularización
def cost_generalizado(thetas, nNeuronas_capa, X, y):
    z,a = FeedForward_generalizado(thetas, nNeuronas_capa, X)
    m = len(X)
    term1 = np.sum(y * np.log(a))
    term2 = np.sum((1 - y) * np.log(1 - a))

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

def backprop_generalizado(thetas, nNeuronas_capa, X, y, lambda_):
    m = len(X)
    z,a = FeedForward_generalizado(thetas, nNeuronas_capa, X)     
    
     # Retropropagación
    deltas = [a[-1] - y]
    #calculo de errores en la capa de salida
    for i in range(len(nNeuronas_capa) - 2, 0, -1):
        delta = np.dot(deltas[0], thetas[i][:, 1:]) * sigmoid_der(z[i - 1])
        deltas.insert(0, delta)

    # Cálculo de gradientes
    gradients = [np.dot(d.T, a[i]) / m for i, d in enumerate(deltas)]
    
    # Regularización de gradientes
    for i in range(len(gradients)):
        gradients[i][:, 1:] += (lambda_ / m) * thetas[i][:, 1:]
    
    cost = reg_cost_generalizado(thetas, nNeuronas_capa, X, y, lambda_)
    
    return cost, gradients


def gradientdescent(theta1, theta2, X, y, lambda_, num_iters, alpha):

    epsilon = 0.12

    theta1 = np.random.uniform(-epsilon, epsilon, size=(theta1.shape[0], theta1.shape[1]))
    theta2 = np.random.uniform(-epsilon, epsilon, size=(theta2.shape[0], theta2.shape[1]))

    for i in range(num_iters):
        J, grad1,grad2=backprop(theta1, theta2, X, y, lambda_)
        
        theta1=theta1-alpha * grad1
        theta2=theta2-alpha * grad2

    return theta1, theta2


def gradientdescent_generalizado(nNeuronas_capa, X, y, lambda_, num_iters, alpha):
    m = len(X)  # Número de ejemplos de entrenamiento

    epsilon = 0.12
    thetas = [np.random.uniform(-epsilon, epsilon, size=(nNeuronas_capa[i + 1], nNeuronas_capa[i] + 1))
              for i in range(len(nNeuronas_capa) - 1)]
    for i, theta in enumerate(thetas):
        print(f"Theta{i + 1} tiene dimensiones: {theta.shape}")
    for iteracion in range(num_iters):
        coste, gradientes = backprop_generalizado(thetas, nNeuronas_capa, X, y, lambda_)

        # Actualizar thetas
        for i in range(len(thetas)):
            thetas[i] -= alpha * gradientes[i]

        if iteracion % 100 == 0:
            print(f"Iteración {iteracion}, Coste: {coste}")

    return thetas