"""
The purpose of this script is to produce plots that show the accuracy of Logistic Regression versus GD iterations
for different learning rates
As instructed in Task 3, all accuracies are estimated using 5-fold cross validation.
"""
import matplotlib.pyplot as plt
import numpy as np
from Project1.src.LogisticRegression import LogisticRegression
from Project1.src.Processor import Processor
from Project1.src.Clean import Clean
from Project1.src.CrossValidation import cross_validation

iters = np.arange(0, 2500, 5)

print("Analyzing the ionosphere data set")
path = "../datasets/ionosphere/ionosphere.data"
header = ["{}{}".format("col", x) for x in range(33 + 1)]
header.append("signal")
All = Processor.read(path, header)
[X, Y] = Clean.Ionosphere(All)

accuracies_1 = []

for iter_ in iters:
    acc, _, _ = cross_validation(5, X.to_numpy(), Processor.ToNumpyCol(Y), LogisticRegression(), learning_rate=0.05,
                                 max_gradient=1e-3, max_iters=iter_, random=False)
    accuracies_1.append(acc)


plt.plot(iters, accuracies_1, 'g-')
plt.ylabel('Accuracy')
plt.xlabel('GD iterations')
plt.legend(['Rate: 0.05'])
plt.show()
plt.savefig('AccVsGDIters.png')
