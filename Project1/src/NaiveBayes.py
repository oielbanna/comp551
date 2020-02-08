import numpy as np


def gaussian_likelihood(x, mean, std):
    return (1 / np.sqrt(2 * np.pi * (std**2))) * (np.exp((-(x-mean)**2 / (2*std**2))))


def posterior(x_test, x_train_split, x, mean, std):
    likelihood = gaussian_likelihood(x_test, mean, std)
    post = np.prod(likelihood, axis=1) * (x_train_split.shape[0] / x.shape[0])
    return post


class NaiveBayes:

    def __init__(self, alpha=1.0):
        """
        This initializes a LogisticRegression object with null weights (empty list).
        """
        self.split = {}
        self.alpha = alpha

    def fit(self, x, y):
        """
        This function trains the model using the given training data and updates the weights of the model accordingly.
        :param x: feature matrix encapsulating data points and the values of their features
        :param y: true class labels of the input data points
        :param learning_rate: gradient descent step
        :param termination: condition for stopping gradient descent
        :return:
        """
        self.x = x
        self.y = y

        #Split binary data into classes 0 and 1
        split = {}
        split[0] = np.array([[]])
        split[1] = np.array([[]])

        one = True
        zero = True
        for i in range(y.shape[0]):
            if(y[i] == 0):
                if(zero == True):
                    split[0] = x[i, :].reshape(x[i, :].shape[0], 1)    #reshape into a column
                    zero = False
                else:
                    split[0] = np.append(split[0], x[i, :].reshape(x[i, :].shape[0], 1), axis=1)   #append column-wise
            elif(y[i] == 1):
                if(one == True):
                    split[1] = x[i, :].reshape(x[i, :].shape[0],1)
                    one = False
                else:
                    split[1] = np.append(split[1], x[i, :].reshape(x[i, :].shape[0], 1), axis=1)

        self.split = split
        self.split[0] = self.split[0].T
        self.split[1] = self.split[1].T

        #self.priors = self.split[0].shape[0]/self.x.shape[0]
        #count = np.array([np.array(i).sum(axis=0) for i in split]) + self.alpha
        #self.feature_prob = count/count.sum(axis=0)[np.newaxis].T

        #Compute means and standard deviations for Gaussian Distribution
        self.mean_one = np.mean(split[1], axis=0)
        self.mean_zero = np.mean(split[0], axis=0)
        self.std_one = np.std(split[1], axis=0)
        self.std_zero = np.std(split[0], axis=0)

        #print(self.std_one)
        #print(self.std_zero)
        #print(self.mean_one)
        #print(self.mean_zero)
        #print(split[0])
        #print(split[1])
        #print(y)

    def predict_probs(self, x_test):
        return [(self.feature_prob * x).sum(axis=1) + self.prior for x in x_test]

    def predict(self, x_test):
        """
        This function predicts the class of the inputted data using the weights of the model object
        :param x: feature matrix encapsulating data points and the values of their features
        :return: predicted labels of the input data points
        """
        post_one = posterior(x_test, self.split[1], self.x, self.mean_one, self.std_one)
        post_zero = posterior(x_test, self.split[0], self.x, self.mean_zero, self.std_zero)

        return 1*(post_one > post_zero)
        #return np.argmax(self.predict_probs(x_test), axis=1)
