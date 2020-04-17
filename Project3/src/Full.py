from scipy.stats import zscore
import numpy as np
from tensorflow.keras import datasets

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images = zscore(train_images, axis=None)
test_images = zscore(test_images, axis=None)

X = train_images
Y = train_labels


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

    def cost(self,
             X,  # N x D
             Y,  # N x C
             W,  # M x C
             V,  # D x M
             ):
        Q = np.dot(X, V)  # N x M
        Z = self.sigmoid(Q)  # N x M
        U = np.dot(Z, W)  # N x K
        print(X.shape, U)
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
        Z = self.sigmoid(np.dot(X, V))  # N x M
        N, D = X.shape
        output = np.dot(Z, W)
        Yh = self.softmax(output)  # N x K

        # cost
        dY = Yh - Y  # N x K

        # backpropagation
        dW = np.dot(Z.T, dY) / N  # M x K
        dZ = np.dot(dY, W.T)  # N x M
        dV = np.dot(X.T, dZ * Z * (1 - Z)) / N  # D x M
        return dW, dV

    def logsumexp(
            self,
            Z,  # NxC
    ):
        Zmax = np.max(Z, axis=1)[:, None]
        lse = Zmax + np.log(np.sum(np.exp(Z - Zmax), axis=1))[:, None]
        return lse  # N

    # def softmax(
    #         self,
    #         u,  # N x C
    # ):
    #     u_exp = np.exp(u - np.max(u)[:, None])
    #     return u_exp / np.sum(u_exp, axis=-1)[:, None]

    def softmax(self, u):
        u_exp = np.exp(u - np.max(u))
        return u_exp / np.sum(u_exp)

    def GD(self, X, Y, M, lr=.1, eps=1e-9, max_iters=10000):
        print(X.shape)
        N, D = X.shape
        N, K = Y.shape
        W = np.random.randn(M, K) * .01
        V = np.random.randn(D, M) * .01
        dW = np.inf * np.ones_like(W)
        t = 0
        while np.linalg.norm(dW) > eps and t < max_iters:
            x_batch, y_batch = batch(X, Y)
            # for x, y in zip(x_batch, y_batch):
            dW, dV = self.gradients(x_batch, y_batch, W, V)
            W = W - lr * dW
            V = V - lr * dV
            t += 1
        return W, V


NN = NN2()
X = np.asarray([x.flatten() for x in X])
Y = np.asarray([OHV(y) for y in Y])
W, V = NN.GD(X, Y, 100)

# print(W, V)
# print(X[100:300].shape)
_, Yhat = NN.cost(X[300:500], Y[300: 500], W, V)
# print(Yhat.shape)
# for img, label in zip(train_images[1:50], train_labels[1:50]):  # testing on 50 images
#     o = NN.predict(img)
#     Yhat = np.append(Yhat, o)
#     # print("Predicted Output: " + str(o))
#     # print("Actual Output: " + str(label))

NN.eval(Yhat, Y[300:500])
