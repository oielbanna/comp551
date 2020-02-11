"""
This file implements scripts for tuning the hyper parameters of the LR models used on the 4 data sets
"""
from Project1.src.LogisticRegression import LogisticRegression
from Project1.src.CrossValidation import cross_validation

dataset = 'ionosphere'

if dataset == 'ionosphere':
    learning_rates = [10, 1, 0.5, 0.1, 0.05, 0.01]

    for rate in learning_rates:
        pass
