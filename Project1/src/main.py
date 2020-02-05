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
        pass

    return accuracy


adult = "./datasets/adult/adult.data"
aheader = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',
           'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'salary']
adultBinaryCols = {
    "sex": {"Male": 0, "Female": 1},
    "salary": {">50K": 0, "<=50K": 1}
}

"""
    **************************************
    SHOWING USAGE OF PROCESSOR CLASS BELOW
"""
# remove last col
X = Processor.read(adult, aheader)
X = Processor.removeMissing(X)
X = Processor.toBinaryCol(X, adultBinaryCols)
Y = X["salary"]
X = X.iloc[:, :-1]
X = Processor.OHE(X)

YHead = Y.head(25).to_numpy()
YHead = YHead.reshape((YHead.shape[0], 1))

model = LogisticRegression()
w = model.fit(X.head(25).to_numpy(), YHead)

X_test = X.tail(5)
Y_test = Y.tail(5)

print(model.predict(X_test.to_numpy()))
# print(Y_test)
