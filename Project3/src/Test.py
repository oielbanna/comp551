import numpy as np
from tensorflow.keras import datasets
from sklearn import metrics

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z + 1e-10))


def eval(Yhat, Ytrue):
    Yhat, Ytrue = Yhat.flatten(), Ytrue.flatten()
    count = np.sum(Yhat == Ytrue)
    print("Accuracy: %f" % ((float(count) / Yhat.shape[0]) * 100))


def softmax(u):
    u_exp = np.exp(u - np.max(u))
    return u_exp / np.sum(u_exp)


def logsumexp(
        Z,  # NxC
):
    Zmax = np.max(Z, axis=1)[:, None]
    lse = Zmax + np.log(np.sum(np.exp(Z - Zmax), axis=1))[:, None]
    return lse  # N


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


def batch(X, Y, batch_size=100):
    randoms = np.random.randint(X.shape[0], size=int(batch_size))
    return X[randoms], Y[randoms]


class NN2:

    def feedforward(self, X, W, V):
        # first layer -> hidden layer
        hidden = np.dot(X, V)
        Z = sigmoid(hidden)  # N x M

        # hidden layer -> output layer
        output = np.dot(Z, W)
        Yh = softmax(output)  # N x K

        return [hidden, output], [Z, Yh]

    def cost(
            self,
            X,  # N x D
            Y,  # N x C
            W,  # M x C
            V,  # D x M
    ):
        Q = np.dot(X, V)  # N x M
        Z = sigmoid(Q)  # N x M
        U = np.dot(Z, W)  # N x K
        Yh = softmax(U)
        nll = - np.mean(np.sum(U * Y, 1) - logsumexp(U))
        return nll

    def gradients(self,
                  X,  # N x D
                  Y,  # N x K
                  W,  # M x K
                  V,  # D x M
                  ):
        Z = sigmoid(np.dot(X, V))  # N x M
        N, D = X.shape
        Yh = softmax(np.dot(Z, W))  # N x K
        dY = Yh - Y  # N x K
        dW = np.dot(Z.T, dY) / N  # M x K
        dZ = np.dot(dY, W.T)  # N x M
        dV = np.dot(X.T, dZ * Z * (1 - Z)) / N  # D x M
        return dW, dV

    def GD(self, X, Y, M, lr=.1, eps=1e-9, max_iters=10000):
        X, Y = batch(X, Y, 64)
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


(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()


def preprocess(X, Y):
    X = np.asarray([x.flatten() for x in X])
    Y = np.asarray([OHV(y) for y in Y])
    return X, Y


NN = NN2()
X, Y = preprocess(train_images, train_labels)
print("Training...")

W, V = NN.GD(X, Y, 50)

X_test, Y_test = preprocess(test_images, test_labels)
# print(X_test.shape)
_, Yhat = NN.feedforward(X_test, W, V)
print(Yhat[-1].shape, Y_test.shape)
eval(np.round(Yhat[-1]), Y_test)
print(round(metrics.accuracy_score(Y_test.flatten(), np.round(Yhat[-1].flatten())), 9))
