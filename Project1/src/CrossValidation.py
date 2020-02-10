import numpy as np
import statistics as stats
from Project1.src.main import evaluate_acc


def cross_validation(k_fold, x, y, model, **kwargs):
    """
    Performs k-fold cross validation with the inputted model's fit() function
    :param k_fold: number of folds to split the data into
    :param x: feature matrix for model training and cross validation
    :param y: labels of the data for model training and cross validation
    :param model: LogisticRegression or NaiveBayes model object
    :param kwargs: arguments taken by the fit function of the model
    :return: mean and standard variation of the validation error on all folds
    """

    # list to hold the accuracy
    accuracy_scores = []

    # Create pseudorandom list of indices for shuffling the input arrays (achieve randomized cross validation)
    shuffle = np.random.RandomState().permutation(len(x))

    # Split the data array into k sub-arrays (folds)
    folds_x = np.array_split(x[shuffle], k_fold)
    folds_y = np.array_split(y[shuffle], k_fold)

    for i in range(len(folds_x)):
        test_x, test_y = folds_x[i], folds_y[i]
        # create the training array by concatenating the remaining k-1 folds
        train_x = np.concatenate([fold for fold in folds_x if fold is not test_x])
        train_y = np.concatenate([fold for fold in folds_y if fold is not test_y])

        model.fit(train_x, train_y, **kwargs)
        y_predicted = model.predict(test_x)
        accuracy_scores.append(evaluate_acc(test_y, y_predicted))

    return [stats.mean(accuracy_scores), stats.stdev(accuracy_scores)]
