import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_der(z):
    return sigmoid(z) * (1 - sigmoid(z))

#Predecir la etiqueta de una entrada dada una red neuronal entrenada.
def predict(pred):   
  
    # Obtener los índices de los valores máximos a lo largo del eje 1
    max_indices = np.argmax(pred, axis=1)
    # Crear un array de ceros con la misma forma que pred
    p = np.zeros_like(pred)
    
    p[np.arange(len(max_indices)),max_indices]=1
    
    return p

#Propagación hacia delante de la red para m capas con n neuronas cada capa
def FeedForward_generalizado(thetas, nNeuronas_capa, X):
    m = len(nNeuronas_capa)
    z = []  # Lista para almacenar las entradas ponderadas
    a = []  # Lista para almacenar las activaciones
    
    #Neuronas de la capa de entrada
    a1= np.array(X)
    a1 = np.hstack([np.ones((a1.shape[0], 1)), a1])
    a.append(a1)
    #neuronas de las capas ocultas
    for i in range (m-2):
        z2 = np.dot(a1, thetas[i].T)
        a2 = sigmoid(z2)
        a2 = np.hstack([np.ones((a2.shape[0], 1)), a2])
        z.append(z2)
        a.append(a2)
        a1 = a2  #Actualizo la capa anterior        

    #neuronas de las capas de salida 
    z3 = np.dot(a2, thetas[m-2].T)
    a3 = sigmoid(z3)
    
    z.append(z3)
    a.append(a3)
    #devuelvo un vector con todas los resultados de las neuronas y otro con los resultados pasados por la activación sigmoide
    return z,a

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

#Coste de la red de neuronas de m numero de capas con n numero de neuronas sin regularización
def cost_generalizado(thetas, nNeuronas_capa, X, y):
    z,a = FeedForward_generalizado(thetas, nNeuronas_capa, X)
    m = len(X)
    
    y_sum = y.values
    term1 = np.sum(y * np.log(a[-1]))
    term2 = np.sum((1 - y) * np.log(1 - a[-1]))
    J = (-1 / m) * np.sum(term1 + term2)
    return J

def backprop_generalizado(thetas, nNeuronas_capa, X, y, lambda_):
    m = len(X)
    z, a= FeedForward_generalizado(thetas, nNeuronas_capa, X)     
    
    #Cálculo de errores en la capa de salida
    deltas = [a[-1] - y]
    
    
    #print("Rango: ",range(len(nNeuronas_capa) - 2, 0, -1))
    for i in range(len(nNeuronas_capa) - 2, 0, -1):
        sigmoid_calc= sigmoid_der(z[i-1])
        delta = np.dot(deltas[0], thetas[i][:, 1:]) * sigmoid_der(z[i-1])
        deltas.insert(0, delta)
    # Cálculo de gradientes
    gradients = [np.dot(d.T, a[i]) / m  for i, d in enumerate(deltas)]
    for g in gradients:
        g = np.hstack([np.ones((g.shape[0], 1)), g])
    # Regularización de gradientes
    for i in range(len(gradients)):
        temp = (lambda_ / m) * thetas[i][:, 1:]        
        gradients[i][:, 1:] += temp
    
    cost = reg_cost_generalizado(thetas, nNeuronas_capa, X, y, lambda_)
    
    return cost, gradients

def gradientdescent_generalizado(nNeuronas_capa, X, y, lambda_, num_iters, alpha):
    m = len(X)
    epsilon = 0.12
    thetas = [np.random.uniform(-epsilon, epsilon, size=(nNeuronas_capa[i + 1], nNeuronas_capa[i] + 1))
              for i in range(len(nNeuronas_capa)-1)]#???
    for iteracion in range(num_iters):
        coste, gradientes = backprop_generalizado(thetas, nNeuronas_capa, X, y, lambda_)

        # Actualizar thetas
        for i in range(len(thetas)):
            thetas[i] -= alpha * gradientes[i]

    return thetas






#Calculos para solo una capa oculta (Practica 5)

#Propagación hacia delante de la red para solo una capa intermedia
def FeedForward(theta1, theta2, X):
    a1 = np.array(X)
    a1 = np.hstack([np.ones((a1.shape[0], 1)), a1])
    z2 = np.dot(a1, theta1.T)
    a2 = sigmoid(z2)
    a2 = np.hstack([np.ones((a2.shape[0], 1)), a2])
    z3 = np.dot(a2, theta2.T)
    a3 = sigmoid(z3)
    return z2,a1, a2, a3

#Cálculo del coste sumando la regularización L2 para una sola capa oculta
def reg_cost(theta1, theta2, X, y, lambda_):
    m = len(X)
    c = cost(theta1,theta2,X,y)
    thetas = np.concatenate((theta1[:,1:].flatten(), theta2[:,1:].flatten()))
    
    J = c + L2(thetas,X,y,lambda_)
    return J

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
