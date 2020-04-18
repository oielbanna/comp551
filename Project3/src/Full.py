from scipy.stats import zscore
import numpy as np
from tensorflow.keras import datasets
import time
from sklearn import metrics


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


def cross_entropy_cost(out2, Y):
    nll = - np.mean(np.sum(out2 * Y, 1) - logsumexp(out2))
    return nll


"""
the following code is adapted from the course slides and changed to fit our data.
"""
class NN2(object):

    def feedforward(self, X, W, V):
        # first layer -> hidden layer
        hidden = np.dot(X, V)
        Z = sigmoid(hidden)  # N x M

        # hidden layer -> output layer
        output = np.dot(Z, W)
        Yh = softmax(output)  # N x K

        return [hidden, output], [Z, Yh]

    def backpropagation(self,
                        X,  # N x D
                        Y,  # N x K
                        W,  # M x K
                        V,  # D x M
                        ):
        # ****** feedforwad stage
        outs, activations = self.feedforward(X, W, V);

        # ****** cost
        dY = activations[-1] - Y  # N x K
        cost = cross_entropy_cost(activations[-1], Y) # calculate cross entropy with softmax output
        if cost > 100:
            print(cost)

        # ****** backpropagation
        N, D = X.shape
        dW = np.dot(activations[0].T, dY) / N  # M x K
        # delta of hidden layer
        dZ = np.dot(dY, W.T)  # N x M
        # delta of weights in the first layer
        dV = np.dot(X.T, dZ * activations[0] * (1 - activations[0])) / N  # D x M
        return dW, dV, cost

    def train(self, X, Y, M, lr=.1, eps=1e-9, epochs=4, batch_size=200, max_iters=10000, verbose=False):
        """
        :param X: Dataset
        :param Y: One hot vector of prediction
        :param M: size of hidden layer
        :param lr: learning rate of gradient descent
        :param eps
        :param epochs
        :param batch_size
        :param max_iters
        :param verbose
        :return:
        """
        # print(X.shape)
        N, D = X.shape
        N, K = Y.shape
        cost_aggregate = []

        # initialize weights
        W = np.random.randn(M, K) * .01
        V = np.random.randn(D, M) * .01
        for epoch in range(epochs):
            dW = np.inf * np.ones_like(W)
            t = 0
            x_batch, y_batch = batch(X, Y, batch_size)
            # mini-batch gradient descent
            while np.linalg.norm(dW) > eps and t < max_iters:
                dW, dV, cost = self.backpropagation(x_batch, y_batch, W, V)

                # update weights
                W = W - lr * dW
                V = V - lr * dV

                t += 1
                cost_aggregate.append(cost)
            if verbose:
                print("Epoch {} with loss {}".format(epoch, np.average(cost_aggregate)))
        return W, V

    def predict(self, X, W, V):
        _, activations = self.feedforward(X, W, V)
        return activations[-1]


def preprocess(X, Y):
    X = np.asarray([x.flatten() for x in X])
    Y = np.asarray([OHV(y) for y in Y])
    return X, Y


(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images = zscore(train_images, axis=None)
test_images = zscore(test_images, axis=None)

NN = NN2()
X, Y = preprocess(train_images, train_labels)
print("Training...")

batchSizes = [64, 200, 1000]
h_sizes = [8, 50, 150]
# start = time.time()
for b in h_sizes:
    W, V = NN.train(X, Y, b, epochs=5, batch_size=64, verbose=True)

    print("Predicting test set...")
    X_test, Y_test = preprocess(test_images, test_labels)
    Yhat = NN.predict(X_test, W, V)
    print(round(metrics.accuracy_score(Y_test, Yhat), 3))
