from scipy.stats import zscore
import numpy as np
from tensorflow.keras import datasets

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images = zscore(train_images, axis=None)
test_images = zscore(test_images, axis=None)


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


class NN2(object):

    def eval(self, Yhat, Ytrue):
        Yhat, Ytrue = Yhat.flatten(), Ytrue.flatten()
        count = np.sum(Yhat == Ytrue)
        print("Accuracy: %f" % ((float(count) / Yhat.shape[0]) * 100))

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z + 1e-10))

    def feedforward(self, X, W, V):
        hidden = np.dot(X, V)
        Z = self.sigmoid(hidden)  # N x M

        output = np.dot(Z, W)
        Yh = self.softmax(output)  # N x K

        return [hidden, output], [Z, Yh]

    def cost(self,
             X,  # N x D
             Y,  # N x C
             W,  # M x C
             V,  # D x M
             ):
        Q = np.dot(X, V)  # N x M
        Z = self.sigmoid(Q)  # N x M
        U = np.dot(Z, W)  # N x K
        # print(X.shape, U)
        Yh = self.softmax(U)
        nll = - np.mean(np.sum(U * Y, 1) - self.logsumexp(U))
        return nll, Yh

    def gradients(self,
                  X,  # N x D
                  Y,  # N x K
                  W,  # M x K
                  V,  # D x M
                  ):
        # feedforward
        # Z = self.sigmoid(np.dot(X, V))  # N x M
        # output = np.dot(Z, W)
        # Yh = self.softmax(output)  # N x K
        outs, activations = self.feedforward(X, W, V);

        # cost
        dY = activations[-1] - Y  # N x K
        cost = self.cost(X, Y, W, V)

        N, D = X.shape
        # backpropagation
        dW = np.dot(activations[0].T, dY) / N  # M x K
        dZ = np.dot(dY, W.T)  # N x M
        dV = np.dot(X.T, dZ * activations[0] * (1 - activations[0])) / N  # D x M
        return dW, dV, cost

    def logsumexp(
            self,
            Z,  # NxC
    ):
        Zmax = np.max(Z, axis=1)[:, None]
        lse = Zmax + np.log(np.sum(np.exp(Z - Zmax), axis=1))[:, None]
        return lse  # N

    def softmax(self, u):
        u_exp = np.exp(u - np.max(u))
        return u_exp / np.sum(u_exp)

    def train(self, X, Y, M, lr=.1, eps=1e-9, epochs=1, max_iters=10000):
        print(X.shape)
        N, D = X.shape
        N, K = Y.shape
        cost_aggregate = []
        for epoch in range(epochs):
            W = np.random.randn(M, K) * .01
            V = np.random.randn(D, M) * .01
            dW = np.inf * np.ones_like(W)
            t = 0
            x_batch, y_batch = batch(X, Y)
            while np.linalg.norm(dW) > eps and t < max_iters:
                dW, dV, cost = self.gradients(x_batch, y_batch, W, V)
                W = W - lr * dW
                V = V - lr * dV
                t += 1
                # print(type(cost))
                # cost_aggregate.append(cost)
            # print("Epoch {} with loss {}".format(epoch, np.average(cost_aggregate)))
        return W, V


def preprocess(X, Y):
    X = np.asarray([x.flatten() for x in X])
    Y = np.asarray([OHV(y) for y in Y])
    return X, Y


NN = NN2()
X, Y = preprocess(train_images, train_labels)

W, V = NN.train(X, Y, 100)

# print(W, V)
# print(X[100:300].shape)
_, Yhat = NN.cost(X[300:500], Y[300: 500], W, V)

NN.eval(Yhat, Y[300:500])
