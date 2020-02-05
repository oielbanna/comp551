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

    def fit(self, x, y, learning_rate=0.01, max_gradient=1e-2, max_iters=np.inf, random_w_low=-1, random_w_high=1):
        """
        This function trains the model using the given training data and updates the weights of the model accordingly.
        :param x: feature matrix encapsulating data points and the values of their features
        :param y: true class labels of the input data points
        :param learning_rate: gradient descent step
        :param max_gradient: max gradient value before gradient descent is stopped
        :param max_iters: maximum number of iterations allowed for gradient descent
        :param random_w_low: lower boundary of the random interval for the weights initialization
        :param random_w_high: higher boundary of the random interval for the weights initialization
        :return: learned weights of the model
        """
        # Initialize the weights array to have as many rows as input features (filled with random values)
        self.weights = np.random.uniform(low=random_w_low, high=random_w_high, size=(x.shape[1], 1))
        g = np.inf
        iterations = 0
        while np.linalg.norm(g) > max_gradient and iterations < max_iters:
            g = gradient(x, y, self.weights)
            self.weights = self.weights - learning_rate * g
            iterations += 1
            # TODO consider plotting the cost vs. iters as the weights are being calculated to check for convergence
        return self.weights

    def predict(self, x):
        """
        This function predicts the class of the inputted data using the weights of the model object
        :param x: feature matrix encapsulating data points and the values of their features
        :return: predicted labels of the input data points
        """
        yh = sigmoid(np.matmul(x, self.weights))
        yh_classes = yh > 0.5  # sets entries to True if > 0.5
        return yh_classes.astype(int)  # returns the predicted labels after transforming True into 1, False into 0

    def __str__(self):
        return "Weights of the model: " + str(self.weights)
