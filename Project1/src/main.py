from Project1.src.LogisticRegression import LogisticRegression
from Project1.src.Processor import Processor
import numpy as np


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


import matplotlib.pyplot as plt

adult = True

if adult:
    path = "./datasets/adult/adult.data"
    header = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
              'relationship',
              'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'salary']
    binaryCols = {
        "sex": {"Male": 0, "Female": 1},
        "salary": {">50K": 0, "<=50K": 1}
    }
    X = Processor.read(path, header)
    X = Processor.removeMissing(X)
    X = Processor.toBinaryCol(X, binaryCols)
    X = X.drop(columns=["capital-gain", "capital-loss"])


    # X['hours-per-week'].value_counts().plot(x='Age', linestyle="None", marker='o')
    print(X['native-country'].value_counts())
    X['native-country'].hist(grid=False)

    # X['marital-status'].value_counts().plot()
    X = Processor.normalize(X, ["fnlwgt", "hours-per-week"])
    Y = X["salary"]
    X = X.iloc[:, :-1]
    # X = Processor.OHE(X)
    plt.waitforbuttonpress()




    [X_train, X_test, Y_train, Y_test] = Processor.split(X, Y, train=0.95)

    # Y_train = Y_train.to_numpy()
    # Y_train = Y_train.reshape((Y_train.shape[0], 1))
    #
    # model = LogisticRegression()
    # w = model.fit(X_train.to_numpy(), Y_train, learning_rate=0.1)
    # print("DONE TRAINING")
    # print(model.predict(X_test.to_numpy()))
    # print(Y_test)

    # TODO frequency distribution of each feature

    # TODO see the correlation between features (pairs) in a scatter plot
    # important features will be seperated

else:
    path = "./datasets/ionosphere/ionosphere.data"
    header = ["{}{}".format("col", x) for x in range(33 + 1)]
    header.append("signal")
    binaryCols = {
        "signal": {"g": 1, "b": 0}
    }

    X = Processor.read(path, header)
    X = Processor.removeMissing(X)
    X = Processor.toBinaryCol(X, binaryCols)
    Y = X["signal"]
    X = X.iloc[:, :-1]
    [X_train, X_test, Y_train, Y_test] = Processor.split(X, Y)

    Y_train = Y_train.to_numpy()
    Y_train = Y_train.reshape((Y_train.shape[0], 1))

    model = LogisticRegression()
    w = model.fit(X_train.to_numpy(), Y_train)

    print(model.predict(X_test.to_numpy()))
    print(Y_test)


