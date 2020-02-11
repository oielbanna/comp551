import numpy as np
import statistics as stats
#from Project1.src.main import evaluate_acc

def evaluate_acc(true_labels, predicted, verbose=False):
    """
    Outputs accuracy score of the model computed from the provided true labels and the predicted ones
    :param true_labels: Numpy array containing true labels
    :param predicted: Numpy array containing labels predicted by a model
    :param verbose: boolean flag, confusion matrix is printed when set to True
    :return: accuracy score
    """
    if true_labels.shape != predicted.shape:
        raise Exception("Input label arrays do not have the same shape.")

    comparison = true_labels == predicted
    correct = np.count_nonzero(comparison)
    accuracy = correct / true_labels.size

    if verbose:
        # Scale predicted labels array by 0.5 and add to comparision array
        # TP -> 1.5, TN -> 1, FP -> 0.5, FN -> 0
        scaled_predicted = 0.5 * predicted
        sum_array = np.add(scaled_predicted, comparison)
        TPs = np.count_nonzero(sum_array == 1.5)
        TNs = np.count_nonzero(sum_array == 1.0)
        FPs = np.count_nonzero(sum_array == 0.5)
        FNs = np.count_nonzero(sum_array == 0)

        confusion_matrix = np.array([[TPs, FPs], [FNs, TNs]])
        precision = TPs / (TPs + FPs)
        recall = TPs / (TPs + FNs)
        print("Confusion Matrix: \n" + str(confusion_matrix))
        print("Precision: " + str(precision))
        print("Recall: " + str(recall))
        print("F1 Score: " + str(2 * (precision * recall) / (precision + recall)))

    return accuracy


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

    ''' # list to hold the accuracy
    accuracy_scores = []

    # Split the data array into k sub-arrays (folds)
    folds_x = np.array_split(x, k_fold)
    folds_y = np.array_split(y, k_fold)'''

    for i in range(len(folds_x)):
        test_x, test_y = folds_x[i], folds_y[i]
        # create the training array by concatenating the remaining k-1 folds
        train_x = np.concatenate([fold for fold in folds_x if fold is not test_x])
        train_y = np.concatenate([fold for fold in folds_y if fold is not test_y])

        model.fit(train_x, train_y, **kwargs)
        #print("DONE")
        y_predicted = model.predict(test_x)
        accuracy_scores.append(evaluate_acc(test_y, y_predicted))

    return [stats.mean(accuracy_scores), stats.stdev(accuracy_scores)]
