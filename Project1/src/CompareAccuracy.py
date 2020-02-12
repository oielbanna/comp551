"""
The purpose of this script is to compare the accuracies of the two models on the 4 data sets and report them in a table.
As instructed in Task 3, all accuracies are estimated using 5-fold cross validation.
Learning rates and threshold gradient were chosen using the results of the hyperparameter tuning script
"""
from Project1.src.LogisticRegression import LogisticRegression
from Project1.src.NaiveBayes import NaiveBayes
from Project1.src.CrossValidation import cross_validation
from Project1.src.Processor import Processor
from Project1.src.Clean import Clean
from Project1.src.HPTuning import df_to_table
import pandas as pd

# Find accuracies for ionosphere data set
print("Analyzing the ionosphere data set")
path = "../datasets/ionosphere/ionosphere.data"
header = ["{}{}".format("col", x) for x in range(33 + 1)]
header.append("signal")
All = Processor.read(path, header)
[X, Y] = Clean.Ionosphere(All)

ionosphere_results = ['ionosphere']
acc, _, _ = cross_validation(5, X.to_numpy(), Processor.ToNumpyCol(Y), LogisticRegression(), learning_rate=1.0,
                             max_gradient=1e-2, max_iters=50000)
ionosphere_results.append(round(acc, 2))
acc = cross_validation(5, X.to_numpy(), Processor.ToNumpyCol(Y), NaiveBayes())
ionosphere_results.append(round(acc, 2))
print(ionosphere_results)

# Find accuracies for adult data set
print("Analyzing the adult data set")
path = "../datasets/adult/adult.data"
header = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',
          'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'salary']
All = Processor.read(path, header)
[X, Y] = Clean.adult(All)
adult_results = ['adult']
acc, _, _ = cross_validation(5, X.to_numpy(), Processor.ToNumpyCol(Y), LogisticRegression(), learning_rate=0.2,
                             max_gradient=1e-1, max_iters=5500)
adult_results.append(round(acc, 2))
acc = cross_validation(5, X.to_numpy(), Processor.ToNumpyCol(Y), NaiveBayes())
adult_results.append(round(acc, 2))
print(adult_results)

# Find accuracies for mammography data set
print("Analyzing the mammography data set")
path = "../datasets/mam/mam.data"
header = ["BI-RADS", "age", "shape", "margin", "density", "result"]
All = Processor.read(path, header)
[X, Y] = Clean.mam(All)
mam_results = ['mam']
acc, _, _ = cross_validation(5, X.to_numpy(), Processor.ToNumpyCol(Y), LogisticRegression(), learning_rate=0.001,
                             max_gradient=1e-1, max_iters=10000)
mam_results.append(round(acc, 2))
acc = cross_validation(5, X.to_numpy(), Processor.ToNumpyCol(Y), NaiveBayes())
mam_results.append(round(acc, 2))
print(mam_results)

# Find the accuracies for tictactoe data set
print("Analyzing the tictactoe data set")
path = "../datasets/tictactoe/tic-tac-toe.data"
header = ["tl", "tm", "tr", "ml", "mm", "mr", "bl", "bm", "br", "result"]
All = Processor.read(path, header)
[X, Y] = Clean.ttt(All)
X = X.astype('float64')
Y = Y.astype('float64')
ttt_results = ['tictactoe']
acc, _, _ = cross_validation(5, X.to_numpy(), Processor.ToNumpyCol(Y), LogisticRegression(), learning_rate=1.0,
                             max_gradient=1e-2, max_iters=100)
ttt_results.append(round(acc, 2))
acc = cross_validation(5, X.to_numpy(), Processor.ToNumpyCol(Y), NaiveBayes())
ttt_results.append(round(acc, 2))
print(ttt_results)

df = pd.DataFrame([ionosphere_results, adult_results, mam_results, ttt_results],
                  columns=['dataset', 'Logistic Regression', 'Naive Bayes'])

df_to_table(df, 'LRvs.NB')
