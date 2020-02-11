import numpy as np


"""Gaussian density function used to calculate likelihood"""
def gaussian_likelihood(x, mean, std):
    return (1 / np.sqrt(2 * np.pi * (std ** 2))) * (np.exp((-(x - mean) ** 2 / (2 * std ** 2))))


def posterior(x_test, x_train_split, x, mean, std):
    """
    Helper function to calculate the posterior probability used in predict
    :param x_test: feature matrix encapsulating data points and the values of their features
    :param x_train_split: the split we are using, either 0 or 1, used to calculate priors
    :param x: the training data used to calculate priors
    :param mean: mean calculated from the data
    :param std: standard deviation calculated from the data
    :return: posterior probabilities
    """

    likelihood = gaussian_likelihood(x_test, mean, std)
    post = np.prod(likelihood, axis=1) * (x_train_split.shape[0] / x.shape[0])  #likelihood multiplied by prior probs
    return post


class NaiveBayes:

    def __init__(self):
        """
        This initializes a NaiveBayes object with empty directory indexed by 0 and 1.
        """
        self.split = {}

    def fit(self, x, y):
        """
        This function trains the model using the given training data by splitting the data based on classes then
        calculating the mean and standard deviation from the data.
        :param x: feature matrix encapsulating data points and the values of their features
        :param y: true class labels of the input data points
        :return:
        """
        self.x = x
        self.y = y

        # Add small epsilon to all data to ensure we dont get any standard deviations of zero
        self.epsilon = 1e-15

        for i in self.x:
            i += self.epsilon

        #Split binary data into classes 0 and 1
        split = {}
        split[0] = np.array([[]])
        split[1] = np.array([[]])

        one = True
        zero = True
        for i in range(self.y.shape[0]):
            if(self.y[i] == 0):
                if(zero == True):
                    split[0] = self.x[i, :].reshape(self.x[i, :].shape[0], 1)    #reshape into a column
                    zero = False
                else:
                    split[0] = np.append(split[0], self.x[i, :].reshape(self.x[i, :].shape[0], 1), axis=1)   #append column-wise
            elif(self.y[i] == 1):
                if(one == True):
                    split[1] = self.x[i, :].reshape(self.x[i, :].shape[0],1)
                    one = False
                else:
                    split[1] = np.append(split[1], self.x[i, :].reshape(self.x[i, :].shape[0], 1), axis=1)

        self.split = split
        self.split[0] = self.split[0].T
        self.split[1] = self.split[1].T

        #Compute means and standard deviations for Gaussian Distribution
        self.mean_one = np.mean(split[1], axis=0)
        self.mean_zero = np.mean(split[0], axis=0)
        self.std_one = np.std(split[1], axis=0)
        self.std_zero = np.std(split[0], axis=0)

        #add epsilon for every mean and standard deviation
        for i in self.mean_zero:
            i += self.epsilon

        for i in self.mean_one:
            i += self.epsilon

        for i in self.std_zero:
            i += self.epsilon

        for i in self.std_one:
            i += self.epsilon

    def predict(self, x_test):
        """
        This function predicts the class of the inputted data using the posterior probabilities of the model object
        :param x_test: feature matrix encapsulating data points and the values of their features
        :return: predicted labels of the input data points
        """
        for x in x_test:
            x += self.epsilon

        #calculate posterior probabilities
        post_one = posterior(x_test, self.split[1], self.x, self.mean_one, self.std_one)
        post_zero = posterior(x_test, self.split[0], self.x, self.mean_zero, self.std_zero)

        #predict labels
        result = 1*(post_one > post_zero)
        result = result.reshape((result.shape[0], 1))

        return result
