import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
import string
import numpy as np


# Get data and convert to numpy array when needed
print('Fetching data...')
#X_train, y_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), return_X_y=True)
#X_train = np.array(X_train)

path = "../datasets/train_reviews.csv"
newsgrous_train = pd.read_csv(path, skipinitialspace=True)

X_train = newsgrous_train['reviews']
y_train = newsgrous_train['target']

norm_vect_train = Cleaner.newsgroups(X_train, subset='train', verbose=True)

tuned_parameters = [{'penalty': ['l1', 'l2'],
                     'loss': ['hinge', 'squared_hinge'],
                     'dual': [True, False],
                     'tol': [0.5, 0.1, 1e-2, 1e-4],
                     'C': [0.05, 0.1, 1.0],
                     'max_iter': [1000, 5000, 10000, 20000]}]

# 5-fold cross validation using an AdaBoost clf with fixed params
print('Cross-validating...')
clf = LinearSVC()
clf = GridSearchCV(clf, tuned_parameters, cv=5, refit=False, verbose=3)
clf.fit(norm_vect_train, y_train)
scores = clf.cv_results_['mean_test_score'].round(3)

print('scores:', scores)
print("The best parameters are %s with a score of %0.3f"
      % (clf.best_params_, clf.best_score_))