import numpy as np


class LogisticRegression:

    def __init__(self, features_number):
        """
        This initializes a LogisticRegression object with null weights (empty list).
        :param features_number: number of features in the feature matrix to initialize weights
        """
        self.weights = np.zeros(features_number)

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
                    reg_param * np.dot(np.transpose(self.weights), self.weights)) / 2.
        # default cost calculation with no regularization
        else:
            return np.mean(y * np.log1p(np.exp(-z)) + (1 - y) * np.log1p(np.exp(z)))

    def fit(self, x, y):

        """
        This function trains the model using the given training data and updates the weights of the model accordingly.
        :param x: feature matrix encapsulating data points and the values of their features
        :param y: true class labels of the input data points
        :return:
        """
        pass

    def predict(self, x):

        """
        This function predicts the class of the inputted data using the weights of the model object
        :param x: feature matrix encapsulating data points and the values of their features
        :return: predicted labels of the input data points
        """
        pass
