import numpy as np

#########################################################################
# NN
#
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

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
    # Transponer para que cada columna sea una muestra
    a1 = X.T
    
     # Añadir sesgo a la primera capa
    a1 = np.vstack([np.ones((1, a1.shape[1])), a1]) 

    z2 = np.dot(theta1, a1)
    a2 = sigmoid(z2)

    # Añadir sesgo a la segunda capa
    a2 = np.vstack([np.ones((1, a2.shape[1])), a2])

    z3 = np.dot(theta2, a2)
    a3 = sigmoid(z3)


    p = (a3 >= 0.5).astype(int)

    return a3.flatten()
