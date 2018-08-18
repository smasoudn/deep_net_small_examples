import numpy as np
from scipy import signal


# Activation functions and their gradients
#############################################
def leaky_relu(x, alpha=0.3):
    return alpha * x if x < 0 else x


def d_leaky_relu(x, alpha=0.3):
    return alpha if x < 0 else 1.0


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))




# Generate inputs
#############################################
def generate_inputs():
    x1 = np.array([[1,1,1,1],
                   [1,0,0,1],
                   [1,0,0,1],
                   [1,0,0,1],
                   [1,1,1,1]])
    x2 = np.array([[1,1,0,0],
                   [0,1,0,0],
                   [0,1,0,0],
                   [0,1,0,0],
                   [0,1,0,0]])
    x3 = np.array([[1,1,1,0],
                   [0,0,1,0],
                   [1,1,1,0],
                   [1,0,0,0],
                   [1,1,1,0]])
    x4 = np.array([ [1,1,1,0],
                    [0,0,1,0],
                    [1,1,1,0],
                    [0,0,1,0],
                    [1,1,1,0]])
    Y = [0.0, 1.0, 2.0, 3.0]
    Y = [Y[i] / len(Y) for i in range(len(Y))]
    X = [x1, x2, x3, x4]

    return X, Y

# Initial weights
#############################################
w1 = np.random.randn(2, 2) * 4
w2 = np.random.randn(1, 1) * 4
w3 = np.random.randn(12, 1) * 4

# Hyper parameter setting
#############################################
num_epochs = 10000
learning_rate = 0.1


# Training
#############################################
X, Y = generate_inputs()

for ep in range(num_epochs):
    total_cost = 0.0
    for i in range(len(X)):
        layer_1 = signal.convolve2d(X[i], w1, 'valid')
        layer_1 = sigmoid(layer_1)

        layer_2 = signal.convolve2d(layer_1, w2, 'valid')
        layer_2 = sigmoid(layer_2)

        layer_2_flattened = np.array([np.reshape(layer_2, -1)])
        layer_3 = layer_2_flattened.dot(w3)
        layer_3 = sigmoid(layer_3)

        cost =  np.square(layer_3 - Y[i]).sum() * 0.5
        total_cost += cost

        delta_3 = (layer_3 - Y[i]) * sigmoid(layer_3) * (1 - sigmoid(layer_3))
        delta_2 = np.reshape(delta_3.dot(w3.T), (4, 3))  * sigmoid(layer_2) * (1 - sigmoid(layer_2))
        delta_1 = delta_2 * w2 * sigmoid(layer_1) * (1  - sigmoid(layer_1))

        del_3 = layer_2_flattened.T.dot(delta_3)
        del_2 = np.rot90(signal.convolve2d(layer_1, np.reshape(delta_2, (4,3)), 'valid'), 2)
        del_1 = np.rot90(signal.convolve2d(X[i], np.rot90(delta_1, 2), 'valid'), 2)

        w3 = w3 - learning_rate * del_3
        w2 = w2 - learning_rate * del_2
        w1 = w1 - learning_rate * del_1
    print(total_cost)


for i in range(len(X)):
    layer_1 = signal.convolve2d(X[i], w1, 'valid')
    layer_1 = sigmoid(layer_1)

    layer_2 = signal.convolve2d(layer_1, w2, 'valid')
    layer_2 = sigmoid(layer_2)

    layer_2_flattened = np.array([np.reshape(layer_2, -1)])
    layer_3 = layer_2_flattened.dot(w3)
    layer_3 = sigmoid(layer_3)

    print(layer_3[0][0] * len(Y), "  ", Y[i] *  len(Y))


