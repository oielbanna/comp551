import numpy as np


def cross_validation(k_fold, data, model, **kwargs):
    """
    Performs k-fold cross validation with the inputted model's fit() function
    :param k_fold: number of folds to split the data into
    :param data: data used for model training and cross validation
    :param model: LogisticRegression or NaiveBayes model object
    :param kwargs: arguments taken by the fit function of the model
    :return: mean and standard variation of the validation error on all folds
    """
