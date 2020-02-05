import numpy as np


def sigmoid(x):
    eps = 1e-9
    return 1 / (1 + np.exp(-x + eps))


def gradient(x, y, w):
    N, D = x.shape
    yh = sigmoid(np.matmul(x, w))
    grad = np.dot(np.transpose(x), yh - y) / N
    return grad


class LogisticRegression:

    def __init__(self):
        """
        This initializes a LogisticRegression object with null weights (empty list).
        """
        self.weights = []

    def compute_cost(self, x, y, regularization=None, reg_param=0):
        """
        This function calculates the cost induced by the current weights of the model
        :param x: feature matrix encapsulating data points and the values of their features
        :param y: true class labels of the input data points
        :param regularization: gives the option to incorporate regularization when fitting the model (L1, L2,
        or no regularization by default)
        :param reg_param: paramter used for regularization
        :return: the cost associated with the current model weights and the training data
        """
        z = np.dot(x, self.weights)
        if regularization == "L1":
            pass
        elif regularization == "L2":
            return np.mean(y * np.log1p(np.exp(-z)) + (1 - y) * np.log1p(np.exp(z))) + (
                    reg_param * np.dot(self.weights, np.transpose(self.weights))) / 2.
        # default cost calculation with no regularization
        else:
            return np.mean(y * np.log1p(np.exp(-z)) + (1 - y) * np.log1p(np.exp(z)))

    def fit(self, x, y, learning_rate=0.01, termination=1e-2):
        """
        This function trains the model using the given training data and updates the weights of the model accordingly.
        :param x: feature matrix encapsulating data points and the values of their features
        :param y: true class labels of the input data points
        :param learning_rate: gradient descent step
        :param termination: condition for stopping gradient descent
        :return:
        """
        # Initialize the weights array to have as many rows as input features (filled with zeros)
        # TODO initialize weights by random sampling, NOT ZEROS
        self.weights = np.ones((x.shape[1], 1))
        g = np.inf
        # TODO change stopping condition to iterations instead of this
        while np.linalg.norm(g) > termination:
            g = gradient(x, y, self.weights)
            self.weights = self.weights - learning_rate * g
            # TODO consider plotting the cost vs. iters as the weights are being calculated to check for convergence
        return self.weights

    def predict(self, x):
        """
        This function predicts the class of the inputted data using the weights of the model object
        :param x: feature matrix encapsulating data points and the values of their features
        :return: predicted labels of the input data points
        """
        yh = sigmoid(np.dot(x, self.weights))
        yh_classes = yh > 0.5  # sets entries to True if > 0.5
        return yh_classes.astype(int)  # returns the predicted labels after transforming True into 1, False into 0

    def __str__(self):
        return "Weights of the model: " + str(self.weights)
