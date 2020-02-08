from Project1.src.LogisticRegression import LogisticRegression
from Project1.src.Processor import Processor
from Project1.src.Clean import Clean
import numpy as np
import matplotlib.pyplot as plt

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

adult = True

if adult:
    path = "./datasets/adult/adult.data"
    header = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
              'relationship',
              'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'salary']

    All = Processor.read(path, header)

    [X, Y] = Clean.adult(All)

    print(X.shape)

    [X_train, X_test, Y_train, Y_test] = Processor.split(X, Y, train=0.8)

    model = LogisticRegression()
    w = model.fit(X_train.to_numpy(), Processor.ToNumpyCol(Y_train), max_gradient=0.8, learning_rate=0.05)

    print("DONE TRAINING")
    print(evaluate_acc(Processor.ToNumpyCol(Y_test), model.predict(X_test.to_numpy())))

else:
    path = "./datasets/ionosphere/ionosphere.data"
    header = ["{}{}".format("col", x) for x in range(33 + 1)]
    header.append("signal")

    All = Processor.read(path, header)

    [X, Y] = Clean.Ionosphere(All)
    [X_train, X_test, Y_train, Y_test] = Processor.split(X, Y, train=0.8)

    model = LogisticRegression()
    w = model.fit(X_train.to_numpy(), Processor.ToNumpyCol(Y_train))

    print(evaluate_acc(Processor.ToNumpyCol(Y_test), model.predict(X_test.to_numpy()), verbose=True))



