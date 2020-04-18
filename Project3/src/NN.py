import numpy as np
from tensorflow.keras import datasets
from sklearn.metrics import accuracy_score


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x):
    return np.maximum(0, x)


def d_relu(x):
    x[x >= 0] = 1
    x[x < 0] = 0
    return x


def softmax(
        u,  # N x C
):
    u_exp = np.exp(u - np.max(u, 1)[:, None])
    return u_exp / np.sum(u_exp, axis=-1)[:, None]


def OHV(vector, size=10):
    """
    Create one hot vector with size = n
    :param size:  size of output vector
    :param vector: assuming vector length = 1
    :return:
    """
    a = np.zeros(size)
    a[vector] = 1
    return a


def preprocess(X, Y):
    """
     For X: flatten the dataset and normalize
     for Y: one hot encode the vector
    """
    X = np.asarray([x.flatten() / 255. for x in X])
    X = X - X.mean(axis=0)
    Y = np.asarray([OHV(y, 10) for y in Y])
    return X, Y


def batch(X, Y, batch_size=100):
    randoms = np.random.choice(X.shape[0], batch_size, replace=True)
    return X[randoms], Y[randoms]


class NN1:
    def __init__(self, activation, d_activation):
        self.activation = activation
        self.d_actication = d_activation

    def gradients(self,
                  X,  # N x D
                  Y,  # N x C
                  W,  # M x C
                  V  # D x M
                  ):
        Q = np.dot(X, V)
        Z = self.activation(Q)  # N x M

        N, D = X.shape
        Yh = self.activation(np.dot(Z, W))  # N x C # N x C

        dY = Yh - Y  # N x C
        dW = np.dot(Z.T, dY) / N  # M x C
        dZ = np.dot(dY, W.T)  # N x M
        dV = np.dot(X.T, dZ * self.d_actication(Q)) / N  # D x M
        return dW, dV

    def GD(self, X, Y, M, batch_size, lr=.1, eps=1e-9, max_iters=10000, verbose=True):
        # init weights
        N, D = X.shape
        N, C = Y.shape
        W = np.random.randn(M, C) * .01
        V = np.random.randn(D, M) * .01

        # init gradient
        dW = np.inf * np.ones_like(W)
        t = 0
        epoch = 0
        epochs = N / batch_size
        while np.linalg.norm(dW) > eps and t < max_iters:
            x, y = batch(X, Y, batch_size)

            # feedforward, backpropagation step
            dW, dV = self.gradients(x, y, W, V)
            W = W - lr * dW
            V = V - lr * dV
            t += 1

            if verbose and (t % epochs) == 0:
                print(str(epoch) + "\t" + str(np.linalg.norm(dW).item()) + "\t" + str(accuracy_score(np.argmax(Y, axis=1), self.predict(X, W, V))))
                epoch += 1

        return W, V

    def predict(self,
                X,
                W,
                V,
                ):
        out1 = np.dot(X, V)
        Z = self.activation(out1)
        out2 = np.dot(Z, W)
        Yh = softmax(out2)
        return np.argmax(Yh, axis=1)


(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
X_train, y_train = preprocess(X_train, y_train)
X_test, y_test = preprocess(X_test, y_test)

NN = NN1(relu, d_relu)
hiddens = [6, 20, 50]
max_iters = [500, 1000, 1500]
# for m in max_iters:
W, V = NN.GD(X_train, y_train, 20, lr=0.01, max_iters=1200, batch_size=1000, verbose=True)
yh = NN.predict(X_test, W, V)
acc = accuracy_score(np.argmax(y_test, axis=1), yh)
print(acc)
