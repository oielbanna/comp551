"""
The purpose of this script is to compare the accuracies of the two models on the 4 data sets and report them in a table.
 As instructed in Task 3, all accuracies are estimated using 5-fold cross validation.
 Learning rates were chosen using the results of the hyperparameter tuning script
"""
from Project1.src.LogisticRegression import LogisticRegression
from Project1.src.NaiveBayes import NaiveBayes
from Project1.src.CrossValidation import cross_validation
from Project1.src.Processor import Processor
from Project1.src.Clean import Clean
from Project1.src.HPTuning import df_to_table

# Find accuracies for ionosphere data set

header = ["{}{}".format("col", x) for x in range(33 + 1)]
header.append("signal")
path = "../datasets/ionosphere/ionosphere.data"
All = Processor.read(path, header)
[X, Y] = Clean.Ionosphere(All)

ionosphere_results = ['ionosphere']
acc, _, _ = cross_validation(5, X.to_numpy(), Processor.ToNumpyCol(Y), LogisticRegression(), learning_rate=0.2,
                             max_gradient=1e-2, max_iters=50000)
ionosphere_results.append(acc)
acc = cross_validation(5, X.to_numpy(), Processor.ToNumpyCol(Y), NaiveBayes())
ionosphere_results.append(acc)

# Find accuracies for adult data set



