from Project1.src.LogisticRegression import LogisticRegression
from Project1.src.NaiveBayes import NaiveBayes
from Project1.src.Processor import Processor
from Project1.src.Clean import Clean
from Project1.src.CrossValidation import cross_validation
import numpy as np
import matplotlib.pyplot as plt


ds = "adult"

if ds == "adult":
    path = "./datasets/adult/adult.data"
    header = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
              'relationship',
              'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'salary']

    All = Processor.read(path, header)

    [X, Y] = Clean.adult(All)

    print(X.shape)

    [X_train, X_test, Y_train, Y_test] = Processor.split(X, Y, train=0.8)

    model = LogisticRegression()

    print(cross_validation(5, X_train.to_numpy(),Processor.ToNumpyCol(Y_train), model, max_gradient=0.8, learning_rate=0.01))


elif ds == "ionosphere":
    path = "./datasets/ionosphere/ionosphere.data"

    header = ["{}{}".format("col", x) for x in range(33 + 1)]
    header.append("signal")

    All = Processor.read(path, header)

    [X, Y] = Clean.Ionosphere(All)

    [X_train, X_test, Y_train, Y_test] = Processor.split(X, Y, train=0.80)

    model = LogisticRegression()

    print(cross_validation(5, X_train.to_numpy(), Processor.ToNumpyCol(Y_train), model))

elif ds == "ttt":
    path = "./datasets/tictactoe/tic-tac-toe.data"
    header = ["tl", "tm", "tr", "ml", "mm", "mr", "bl", "bm", "br", "result"]




