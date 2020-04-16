import numpy as np
from tensorflow.keras import datasets


def softmax(u):
    u_exp = np.exp(u - np.max(u))
    return u_exp / np.sum(u_exp)


def d_softmax(z):
    return - softmax(z) * softmax(z)


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


def CrossEntropyLoss(yHat, y):
    L = -1 / len(y) * np.sum(np.log(yHat + 1e-15))
    return L
    # return - np.sum(y * np.log(yHat + 1e-15))


def d_crossEntropyLoss(yhat, y):
    # print(np.sum(yhat), np.sum(y))
    d = - np.sum(np.divide(y + 1e-15, yhat + 1e-15) + np.log(yhat + 1e-15))
    # d = np.sum(- np.divide(y, yhat), np.divide((1-y), 1-yhat)) / y.shape[0]
    # print("dasiof", d)
    if d != d:
        print("Shit something bad happened")
        exit(5)
    return d


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z + 1e-10))


def delta_sigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))


def batch(X, Y, batch_size=100):
    randoms = np.random.randint(X.shape[0], size=int(batch_size))
    return X[randoms], Y[randoms]


class NN2(object):
    def __init__(self, hidden_size=50):
        self.inputSize = 3072
        self.outputSize = 10
        self.hiddenSize = hidden_size

        np.random.seed(5)
        # weights
        self.W = np.random.randn(self.outputSize, self.inputSize) * 0.001 # input to hidden
        # self.V = np.random.randn(self.hiddenSize, self.outputSize)  # hidden to output
        # print(self.W.shape, self.V.shape)

    def eval(self, Yhat, Ytrue):
        print(Yhat, Ytrue)
        count = np.sum(Yhat == Ytrue)
        print("Accuracy: %f" % ((float(count) / Yhat.shape[0]) * 100))

    def train(self, X, Y, epochs=5, lr=1e-5, batch_size=100):
        cost_aggregate = []
        for epoch in range(epochs):
            x, y_t = batch(X, Y, batch_size)
            x = np.asarray([e.flatten() for e in x])
            # x = x.flatten()

            # Calculate loss and gradient for the iteration
            loss, grad = self.cross_entropy_loss(x, y_t, 1e-3)
            cost_aggregate.append(loss)

            # delta_z2_error, delta_cost = self.backpropagation(Yhat, ohv_yt, outs)

            self.W -= lr * grad

            print("Epoch {} with loss {}".format(epoch, np.average(cost_aggregate)))

    def predict(self, x):
        """
        Predict labels using the trained model.
        Arguments:
            x: D * N numpy array as the test data, where D is the dimension and N the test sample size
        Output:
            y_pred: 1D numpy array with length N as the predicted labels for the test data
        """
        y = self.W.dot(x)
        y_pred = np.argmax(y, axis=0)
        return y_pred

    def cross_entropy_loss(self, x, y, reg):
        """
        Calculate the cross-entropy loss and the gradient for each iteration of training.
        Arguments:
            x: D * N numpy array as the training data, where D is the dimension and N the training sample size
            y: 1D numpy array with length N as the labels for the training data
        Output:
            loss: a float number of calculated cross-entropy loss
            dW: C * D numpy array as the calculated gradient for W, where C is the number of classes, and 10 for this model
        """

        # Calculation of loss
        z = np.matmul([self.W] * x.shape[0], x.T)
        z -= np.max(z, axis=0)  # Max trick for the softmax, preventing infinite values
        p = np.exp(z) / np.sum(np.exp(z), axis=0)  # Softmax function
        print(np.sum(p), y.shape)
        L = -1 / len(y) * np.sum(np.log(p[y, range(len(y) - 1)]))  # Cross-entropy loss
        R = 0.5 * np.sum(np.multiply(self.W, self.W))  # Regularization term
        loss = L + R * reg  # Total loss

        # Calculation of dW
        p[y, range(len(y))] -= 1
        dW = 1 / len(y) * p.dot(x.T) + reg * self.W
        return loss, dW


(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
X = train_images / 255.0
Y = train_labels

NN = NN2()
NN.train(X[0:1000], Y[0:1000])
Yhat = np.array([])
for img, label in zip(test_images[1:10], test_labels[1:10]):  # testing on 10 images
    o = NN.predict(img)
    Yhat = np.append(Yhat, o)
    # print("Predicted Output: " + str(o))
    # print("Actual Output: " + str(label))

NN.eval(Yhat, test_labels[1:10])
