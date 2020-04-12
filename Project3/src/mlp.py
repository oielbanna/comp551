import numpy as np
import torch


def relu(z):
    if np.isscalar(z):
        result = np.max((z, 0))
    else:
        zero_aux = np.zeros(z.shape)
        meta_z = np.stack((z, zero_aux), axis=-1)
        result = np.max(meta_z, axis=-1)
    return result


def logsumexp(Z): # NxC
    Zmax = np.max(Z, axis=1)[:, None]
    lse = Zmax + np.log(np.sum(np.exp(Z - Zmax), axis=1))[:, None]
    return lse  # N


def softmax(u): # N x C
    u_exp = np.exp(u - np.max(u, 1)[:, None])
    return u_exp / np.sum(u_exp, axis=-1)[:, None]

def sigmoid(z):
    result = 1.0 / (1.0 + np.exp(-z))
    return result

def sigmoid_derivative(z):
    result = sigmoid(z) * (1 - sigmoid(z))
    return result

def relu_derivative(z):
    result = 1 * (z>0)
    return result

class MLP:

    def __init__(self, size_per_layer, activation_func):
        self.layer_size = size_per_layer
        self.n_layers = len(size_per_layer)
        self.activation_func = activation_func

        self.weights = self.initialize_weights()

    def train(self, X, Y):
        pass

    def predict(self, X):
        pass

    def initialize_weights(self):
        weights = []
        for i in range(0, self.n_layers - 1):
            size_layer = self.layer_size[i]
            size_next_layer = self.layer_size[i + 1]
            if self.activation_func == 'sigmoid':
                theta_tmp = ((np.random.rand(size_next_layer, size_layer) * 2.0) - 1)
            elif self.activation_func == 'relu':
                theta_tmp = (np.random.rand(size_next_layer, size_layer))
            weights.append(theta_tmp)
        return np.asarray(weights)

    def GD(self, X, Y, M, learning_rate=.1, eps=1e-9, max_iters=100000):
        N, D = X.shape
        N, K = Y.shape

        W = np.random.randn(M, K) * .01
        V = np.random.randn(D, M) * .01
        dW = np.inf * np.ones_like(W)
        t = 0
        while np.linalg.norm(dW) > eps and t < max_iters:
            dW, dV = self.gradients(X, Y, W, V)
            W = W - learning_rate * dW
            V = V - learning_rate * dV
            t += 1
        return W, V

    def gradients(self,
                  X,  # N x D
                  Y,  # N x K
                  W,  # M x K
                  V,  # D x M
                  ):
        Z = self.logistic(np.dot(X, V))  # N x M
        N, D = X.shape
        Yh = softmax(np.dot(Z, W))  # N x K
        dY = Yh - Y  # N x K
        dW = np.dot(Z.T, dY) / N  # M x K
        dZ = np.dot(dY, W.T)  # N x M
        dV = np.dot(X.T, dZ * Z * (1 - Z)) / N  # D x M
        return dW, dV

    def feedforward(self, x):
        input_layer = x
        a = self.n_layers * [None]
        z = self.n_layers * [None]
        output_layer = 0.0

        for layer in range(self.n_layers-1):
            a[layer] = input_layer
            z[layer+1] = np.dot(input_layer, self.weights[layer].tranpose())

            if self.activation_func == 'sigmoid':
                #if last layer, apply softmax
                if(layer + 1 == self.n_layers-1):
                    output_layer = softmax(z[layer+1])
                else:
                    output_layer = sigmoid(z[layer+1])
            elif self.activation_func == 'relu':
                # if last layer, apply softmax
                if (layer + 1 == self.n_layers - 1):
                    output_layer = softmax(z[layer + 1])
                else:
                    output_layer = relu(z[layer + 1])

            input_layer = output_layer

        a[self.n_layers-1] = output_layer
        return a, z

    def backpropagation(self, x, y):
        n_examples = x.shape[0]
        a, z = self.feedforward(y)

        #backpropagation, calculate weights based on the softmax function (last layer)
        updates = self.n_layers * [None]
        updates[-1] = a[-1] - y

        #update weights from the second last layer to the second layer
        for layer in np.arrange(self.n_layers-2, 0, -1):
            if self.activation_func == 'sigmoid':
                updates[layer] = (np.dot(self.weights[layer].T, updates[layer+1].T).T) * sigmoid_derivative(z[layer])
            elif self.activation_func == 'relu':
                updates[layer] = (np.dot(self.weights[layer].T, updates[layer+1].T).T) * relu_derivative(z[layer])

        #compute gradient
        gradients = (self.n_layer - 1) * [None]
        for layer in range(self.n_layers - 1):
            temp_grads = np.dot(updates[layer+1].T, a[layer]) / n_examples
            temp_grads = temp_grads + (self.lambda_r / n_examples) * self.weights[layer][:, 1:]

            gradients[layer] = temp_grads

        return gradients
