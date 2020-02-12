"""
The purpose of this script is to produce plots that show the accuracy of Logistic Regression versus GD iterations
for different learning rates
As instructed in Task 3, all accuracies are estimated using 5-fold cross validation.
"""
import matplotlib.pyplot as plt
import numpy as np
from Project1.src.LogisticRegression import LogisticRegression
from Project1.src.NaiveBayes import NaiveBayes
from Project1.src.Processor import Processor
from Project1.src.Clean import Clean
from Project1.src.CrossValidation import cross_validation
from Project1.src.CrossValidation import evaluate_acc



print("Analyzing the ionosphere data set")
path = "../datasets/ionosphere/ionosphere.data"
header = ["{}{}".format("col", x) for x in range(33 + 1)]
header.append("signal")
All = Processor.read(path, header)
[X, Y] = Clean.Ionosphere(All)

X = X.to_numpy()
Y = Processor.ToNumpyCol(Y)

iters = np.arange(20, X.shape[0], 50)
#print(X.shape)
#print(Y.shape)

accuracies = []

for iter_ in iters:
    #rowsX = X[0:X.shape[0], :]
    #rowsY = Y[0:Y.shape[0], :]
    rowsX = X[0:iter_, :]
    rowsY = Y[0:iter_, :]
    #acc, _, _ = cross_validation(5, rowsX, rowsY, LogisticRegression(), learning_rate=0.1, max_gradient=1e-3, max_iters=iter_)
    acc = cross_validation(5, rowsX, rowsY, NaiveBayes())
    accuracies.append(acc)

"""path = "../datasets/ionosphere/ionosphere.data"
header = ["{}{}".format("col", x) for x in range(33 + 1)]
header.append("signal")
All = Processor.read(path, header)
[X, Y] = Clean.Ionosphere(All)

path = "../datasets/adult/adult.data"

header = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
          'relationship',
          'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'salary']

All = Processor.read(path, header)

[X, Y] = Clean.adult(All)

path = "../datasets/mam/mam.data"
header = ["BI-RADS", "age", "shape", "margin", "density", "result"]
All = Processor.read(path, header)

[X, Y] = Clean.mam(All)"""

path = "../datasets/tictactoe/tic-tac-toe.data"
header = ["tl", "tm", "tr", "ml", "mm", "mr", "bl", "bm", "br", "result"]

All = Processor.read(path, header)

[X, Y] = Clean.ttt(All)

X = X.to_numpy()
Y = Processor.ToNumpyCol(Y)

iters = np.arange(100, X.shape[0], 100)
#print(X.shape)
#print(Y.shape)

accuraciesNB = []
accuraciesLR = []

for iter_ in iters:
    #rowsX = X[0:X.shape[0], :]
    #rowsY = Y[0:Y.shape[0], :]
    rowsX = X[0:iter_, :]
    rowsY = Y[0:iter_, :]
    rowsX = rowsX.astype('float64')
    rowsY = rowsY.astype('float64')
    #acc, _, _ = cross_validation(5, rowsX, rowsY, LogisticRegression(), learning_rate=0.1, max_gradient=1e-3, max_iters=iter_)
    acc = cross_validation(5, rowsX, rowsY, NaiveBayes())
    accuraciesNB.append(acc)

for iter_ in iters:
    #rowsX = X[0:X.shape[0], :]
    #rowsY = Y[0:Y.shape[0], :]
    rowsX = X[0:iter_, :]
    rowsY = Y[0:iter_, :]
    acc, _, _ = cross_validation(5, rowsX, rowsY, LogisticRegression(), learning_rate=0.1, max_gradient=1e-3, max_iters=iter_)
    #acc = cross_validation(5, rowsX, rowsY, NaiveBayes())
    accuraciesLR.append(acc)

plt.plot(iters, accuraciesNB, 'b')
plt.plot(iters, accuraciesLR, 'g')

plt.ylabel('Accuracy')
plt.xlabel('Train Size (Rows)')
plt.legend(['Naive Bayes', 'LogisticRegression'])
plt.show()
plt.savefig('AccVsTrainSizeMam.png')