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
        self.W = np.random.randn(self.inputSize, self.hiddenSize)  # input to hidden
        self.V = np.random.randn(self.hiddenSize, self.outputSize)  # hidden to output
        print(self.W.shape, self.V.shape)

    def eval(self, Yhat, Ytrue):
        print(Yhat, Ytrue)
        count = np.sum(Yhat == Ytrue)
        print("Accuracy: %f" % ((float(count) / Yhat.shape[0]) * 100))

    def feedforward(self, X):
        # inputs to layer 1
        # print(self.W)
        out = np.dot(X.flatten(), self.W)  # 3027x1 * 3027x50 => 50x1
        activation = sigmoid(out)

        # layer 1 to output
        out2 = np.dot(activation, self.V)  # 50x1 * 50x10 => 10x1
        out2 -= np.max(out2, axis=0)  # Max trick for the softmax, preventing infinite values
        activation2 = softmax(out2)

        return activation2, [out, out2], [activation, activation2]

    def backpropagation(self, yhat, ohv_yt, outs):
        # TODO multply by deriv of cost (cross entropy), not cost on its own
        # TODO delta_sigmoid here should be the delta for sofrtmax, not sigmoid so we have to change that too
        delta_z1_cost = d_crossEntropyLoss(yhat, ohv_yt) * d_softmax(outs[1])  # delta cost for second to last layer

        z2_error = np.dot(delta_z1_cost, self.V.T)

        delta_z2_error = z2_error * delta_sigmoid(outs[0])  # delta cost for first layer

        return delta_z2_error, delta_z1_cost

    def train(self, X, Y, epochs=5, lr=1e-5, batch_size=10000):
        cost_aggregate = []
        for epoch in range(epochs):
            X_batch, Y_batch = batch(X, Y, batch_size)
            
            for x, y_t in zip(X_batch, Y_batch):
                x = x.flatten()
                Yhat, outs, activations = self.feedforward(x)
                # print("Is this Yhat? ", Yhat)
                # TODO make one hot vector from y_t and take the negative log of the activations[1] and do dot product between them.
                # - np.sum(y_t * log(predicted))
                ohv_yt = OHV(y_t)
                cost = CrossEntropyLoss(Yhat, ohv_yt)
                # print("COST ",cost)
                # cost = y_t[0] - Yhat  # => cost is a scalar
                cost_aggregate.append(cost)

                delta_z2_error, delta_cost = self.backpropagation(Yhat, ohv_yt, outs)

                self.W += np.dot(x.reshape(-1, 1), delta_z2_error.reshape(-1, 1).T) * lr  # input to hidden weights
                self.V += np.dot(activations[0].reshape(-1, 1),
                                 delta_cost.reshape(-1, 1).T) * lr  # hidden tp output weights

            print("Epoch {} with loss {}".format(epoch, np.average(cost_aggregate)))

    def predict(self, X):
        Yhat, _, _ = self.feedforward(X)
        return Yhat


(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
X = train_images / 255.0
Y = train_labels

NN = NN2()
NN.train(X, Y)
Yhat = np.array([])
for img, label in zip(test_images[1:10], test_labels[1:10]):  # testing on 10 images
    o = NN.predict(img)
    Yhat = np.append(Yhat, o)
    # print("Predicted Output: " + str(o))
    # print("Actual Output: " + str(label))

NN.eval(Yhat, test_labels[1:10])
