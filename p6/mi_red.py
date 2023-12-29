import numpy as np

#función sigmoide de activación
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#derivada de la sigmoide
def sigmoid_der(z):
    return sigmoid(z) * (1 - sigmoid(z))

#propagación hacia delante 
def FeedForward(X, thetas):
    activations = [X]
    for theta in thetas:
        X = np.insert(X,0,1, axis=1)#añado el término independiente
        X = sigmoid(np.dot(X, theta.T))
        activations.append(X)
    return activations

#inicializa la matriz de thetas con valores aleatorios
def initThetas(n_nodes):
    thetas = []
    epsilon = 0.12    
    for i in range(1, len(n_nodes)):
        theta_capa = np.random.uniform(-epsilon, epsilon, size=(n_nodes[i], n_nodes[i-1] + 1))
        thetas.append(theta_capa)
    return thetas

#cálculo del error final
def calcula_dif(y_pred, y_true):
    return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) / len(y_true)

def backpropagation(X, y, n_nodes, weights, learning_rate=0.01, epochs=1000):
    m = len(X)
    
    for epoch in range(epochs):
        # Forward propagation
        activations = FeedForward(X, weights)
        output = activations[-1]

        # Backward propagation
        delta = [output - y]
        for i in range(len(activations) - 2, 0, -1):
            delta_i = np.dot(delta[0], weights[i][:, 1:]) * sigmoid_der(np.dot(activations[i-1],weights[i][:, 1:]))
            delta.insert(0, delta_i)

        # Update weights
        for i in range(len(weights)):
            weights[i] -= learning_rate * np.dot(delta[i].T, np.insert(activations[i], 0, 1, axis=1)) / m

        # Print loss every 100 epochs
        if epoch % 100 == 0:
            loss = calcula_dif(output, y)
            print(f"Epoch {epoch}, Loss: {loss}")

    return weights