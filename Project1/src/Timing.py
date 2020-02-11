from Project1.src.LogisticRegression import LogisticRegression
from Project1.src.NaiveBayes import NaiveBayes
from Project1.src.Processor import Processor
from Project1.src.Clean import Clean
from Project1.src.CrossValidation import cross_validation
import timeit
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def df_to_table(pandas_frame, export_filename):
    fig, ax = plt.subplots()

    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    ax.table(cellText=pandas_frame.values, colLabels=pandas_frame.columns, loc='center')

    fig.tight_layout()

    plt.savefig(export_filename + '.png', bbox_inches='tight')

ds = "ionosphere"

if ds == "adult":
    """NAIVE BAYES
    path = "../datasets/adult/adult.data"

    header = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
              'relationship',
              'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'salary']

    All = Processor.read(path, header)

    [X, Y] = Clean.adult(All)

    print(X.shape)

    [X_train, X_test, Y_train, Y_test] = Processor.split(X, Y, train=0.8)

    model = NaiveBayes()
    w = model.fit(X_train.to_numpy(), Processor.ToNumpyCol(Y_train))"""

    """NAIVEBAYES IONOSPHERE"""

    setup_NB = '''
from Project1.src.NaiveBayes import NaiveBayes
from Project1.src.LogisticRegression import LogisticRegression
from Project1.src.Processor import Processor
from Project1.src.Clean import Clean
from Project1.src.CrossValidation import cross_validation

path = "../datasets/ionosphere/ionosphere.data"

header = ["{}{}".format("col", x) for x in range(33 + 1)]
header.append("signal")

All = Processor.read(path, header)

[X, Y] = Clean.Ionosphere(All)

[X_train, X_test, Y_train, Y_test] = Processor.split(X, Y, train=0.8)

model = NaiveBayes()
    '''

    setup_LR = '''
from Project1.src.NaiveBayes import NaiveBayes
from Project1.src.LogisticRegression import LogisticRegression
from Project1.src.Processor import Processor
from Project1.src.Clean import Clean
from Project1.src.CrossValidation import cross_validation

path = "../datasets/ionosphere/ionosphere.data"

header = ["{}{}".format("col", x) for x in range(33 + 1)]
header.append("signal")

All = Processor.read(path, header)

[X, Y] = Clean.Ionosphere(All)

[X_train, X_test, Y_train, Y_test] = Processor.split(X, Y, train=0.8)

model = LogisticRegression()
    '''

    test_NB = '''
w = model.fit(X_train.to_numpy(), Processor.ToNumpyCol(Y_train))
    '''

    test_LR = '''
w = model.fit(X_train.to_numpy(), Processor.ToNumpyCol(Y_train), learning_rate=0.2, max_gradient=1e-2)
'''

    timeNB = timeit.timeit(setup=setup_NB, stmt=test_NB, number=10000) / 10000
    timeLR = timeit.timeit(setup=setup_LR, stmt=test_LR, number=10000) / 10000
    result = []
    result.append("Ionosphere")
    result.append(timeNB)
    result.append(timeLR)

    df = pd.DataFrame(result, columns=['dataset', 'execution time (Naive Bayes)', 'execution time (Logistic Regression)'])
    df_to_table(df, 'ionosphere_time')

    # print(evaluate_acc(Processor.ToNumpyCol(Y_test), model.predict(X_test.to_numpy())))

    #print(cross_validation(5, X_train.to_numpy(), Processor.ToNumpyCol(Y_train), model))

elif ds == "ionosphere":
    path = "../datasets/ionosphere/ionosphere.data"

    header = ["{}{}".format("col", x) for x in range(33 + 1)]
    header.append("signal")

    All = Processor.read(path, header)

    [X, Y] = Clean.Ionosphere(All)

    [X_train, X_test, Y_train, Y_test] = Processor.split(X, Y, train=0.8)
    setup = '''
from Project1.src.NaiveBayes import NaiveBayes
from Project1.src.Processor import Processor
from Project1.src.Clean import Clean
from Project1.src.CrossValidation import cross_validation

path = "../datasets/ionosphere/ionosphere.data"

header = ["{}{}".format("col", x) for x in range(33 + 1)]
header.append("signal")

All = Processor.read(path, header)

[X, Y] = Clean.Ionosphere(All)

[X_train, X_test, Y_train, Y_test] = Processor.split(X, Y, train=0.8)

model = NaiveBayes()
'''

    test = '''
w = model.fit(X_train.to_numpy(), Processor.ToNumpyCol(Y_train))
'''

    time = timeit.timeit(setup=setup, stmt=test, number=10000) / 10000
    print(time)

    #print(cross_validation(5, X_train.to_numpy(), Processor.ToNumpyCol(Y_train), model))

elif ds == "mam":
    path = "./datasets/mam/mam.data"
    header = ["BI-RADS", "age", "shape", "margin", "density", "result"]
    All = Processor.read(path, header)

    [X, Y] = Clean.mam(All)

    [X_train, X_test, Y_train, Y_test] = Processor.split(X, Y, train=0.8)

    model = NaiveBayes()

    print(cross_validation(5, X_train.to_numpy(), Processor.ToNumpyCol(Y_train), model))

elif ds == "ttt":
    path = "./datasets/tictactoe/tic-tac-toe.data"
    header = ["tl", "tm", "tr", "ml", "mm", "mr", "bl", "bm", "br", "result"]

    All = Processor.read(path, header)

    [X, Y] = Clean.ttt(All)

    print(X.shape)

    [X_train, X_test, Y_train, Y_test] = Processor.split(X, Y, train=0.8)

    model = NaiveBayes()
    if type(model) == NaiveBayes:
        X_train = X_train.astype('float64')
        Y_train = Y_train.astype('float64')

    print(cross_validation(5, X_train.to_numpy(), Processor.ToNumpyCol(Y_train), model))