"""
This file implements scripts for tuning the hyper parameters of the LR models used on the 4 data sets
"""
from Project1.src.LogisticRegression import LogisticRegression
from Project1.src.CrossValidation import cross_validation
from Project1.src.Processor import Processor
from Project1.src.Clean import Clean
import pandas as pd
import matplotlib.pyplot as plt


def df_to_table(pandas_frame, export_filename):
    fig, ax = plt.subplots()

    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    ax.table(cellText=pandas_frame.values, colLabels=pandas_frame.columns, loc='center')

    fig.tight_layout()

    plt.savefig(export_filename + '.png', bbox_inches='tight')


dataset = 'adult'

if dataset == 'ionosphere':
    path = "../datasets/ionosphere/ionosphere.data"

    header = ["{}{}".format("col", x) for x in range(33 + 1)]
    header.append("signal")

    All = Processor.read(path, header)

    [X, Y] = Clean.Ionosphere(All)

    [X_train, X_test, Y_train, Y_test] = Processor.split(X, Y, train=0.8)

    learning_rates = [10, 5, 1, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.01]

    results = []

    for rate in learning_rates:
        r = cross_validation(5, X_train.to_numpy(), Processor.ToNumpyCol(Y_train), LogisticRegression(),
                             learning_rate=rate, max_gradient=1e-2, max_iters=50000, random=True)
        r.insert(0, rate)
        results.append(r)

    df = pd.DataFrame(results, columns=['learning rate', 'accuracy', 'last gradient', 'iterations'])
    df_to_table(df, 'ionosphere_table')

elif dataset == 'adult':
    path = "../datasets/adult/adult.data"

    header = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
              'relationship',
              'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'salary']

    All = Processor.read(path, header)

    [X, Y] = Clean.adult(All)

    print(X.shape)

    [X_train, X_test, Y_train, Y_test] = Processor.split(X, Y, train=0.8)

    learning_rates = [0.2, 0.1]

    results = []

    for rate in learning_rates:
        r = cross_validation(5, X_train.to_numpy(), Processor.ToNumpyCol(Y_train), LogisticRegression(),
                             learning_rate=rate, max_gradient=1e-2, max_iters=10000, random=True)
        r.insert(0, rate)
        results.append(r)

    df = pd.DataFrame(results, columns=['learning rate', 'accuracy', 'last gradient', 'iterations'])
    print(df)
    # df_to_table(df, 'adult_table')
