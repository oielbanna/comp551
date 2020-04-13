import numpy as np
from tensorflow.keras import datasets


def softmax(u):  # N x C
    print(u.shape)
    u_exp = np.exp(u - np.max(u, 1)[:, None])
    return u_exp / np.sum(u_exp, axis=-1)[:, None]


def SSD(Yh, Y):
    return np.sum((Yh - Y) ** 2)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def deriv_sigmoid(s):
    # derivative of sigmoid
    return s * (1 - s)


def batch(X, Y, batch_size=100):
    randoms = np.random.randint(X.shape[0], size=int(batch_size))
    return X[randoms], Y[randoms]


class NN2(object):
    def __init__(self, hidden_size=50):
        self.inputSize = 3072
        self.outputSize = 10
        self.hiddenSize = hidden_size

        # weights
        self.W = np.random.randn(self.inputSize, self.hiddenSize)  # input to hidden
        self.V = np.random.randn(self.hiddenSize, self.outputSize)  # hidden to output
        print(self.W.shape, self.V.shape)

    def feedforward(self, X):
        # inputs to layer 1
        out = np.dot(X.flatten(), self.W)  # 3027x1 * 3027x50
        activation = sigmoid(out)

        # layer 1 to output
        out2 = np.dot(activation, self.V)  # TODO ??? * 50x10
        Yhat = sigmoid(out2)

        return Yhat, [out, out2], [activation, Yhat]

    def backpropagation(self, Yhat, cost, activations):
        delta_cost = cost * deriv_sigmoid(Yhat)  # delta cost for second to last layer

        z2_error = np.dot(delta_cost, self.V.T)

        delta_z2_error = z2_error * deriv_sigmoid(activations[0])  # delta cost for first layer

        return delta_z2_error, delta_cost

    def train(self, X, Y, epochs=1000, lr=0.1, batch_size=100):
        n_batches = int(X.shape[0] / batch_size)

        for epoch in range(epochs):
            for b in range(n_batches):
                X_batch, Y_batch = batch(X, Y, batch_size)
                for x, y_t in zip(X_batch, Y_batch):
                    x = x.flatten()
                    Yhat, outs, activations = self.feedforward(x)

                    # cost = SSD(y_t, Yhat)  # sum of square difference loss
                    cost = y_t - Yhat

                    delta_z2_error, delta_cost = self.backpropagation(Yhat, cost, activations)

                    print(delta_z2_error.shape)
                    # update the weights
                    self.W += np.dot(x, delta_z2_error.T) * lr  # adjusting first set (input --> hidden) weights
                    self.V += activations[0].T.dot(delta_cost) * lr  # adjusting second set (hidden --> output) weights

    def predict(self, X):
        Yhat, _, _ = self.feedforward(X)
        return Yhat


(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
X = train_images / 255.0
Y = train_labels

NN = NN2()
NN.train(X, Y)
o = NN.feedforward(test_images)
print("Predicted Output: \n" + str(o))
print("Actual Output: \n" + str(test_labels))
