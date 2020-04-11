import numpy as np


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


class MLP:

    def __init__(self, size_per_layer, activation_func):
        self.layer_size = size_per_layer
        self.n_layers = len(size_per_layer)
        self.activation_func = activation_func

        self.initialize_weights()

    def train(self, X, Y):
        pass

    def predict(self, X):
        pass

    def initialize_weights(self):
        pass

    def GD(self, X, Y, M, lr=.1, eps=1e-9, max_iters=100000):
        N, D = X.shape
        N, K = Y.shape

        W = np.random.randn(M, K) * .01
        V = np.random.randn(D, M) * .01
        dW = np.inf * np.ones_like(W)
        t = 0
        while np.linalg.norm(dW) > eps and t < max_iters:
            dW, dV = self.gradients(X, Y, W, V)
            W = W - lr * dW
            V = V - lr * dV
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
