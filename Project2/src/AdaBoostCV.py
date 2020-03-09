"""
This script is an example of how we tune hyperparameters by using the GridSearchCV model from Sklearn.
"""
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from Project2.src.Cleaner import Cleaner

# Get data and convert to numpy array when needed
print('Fetching data...')
X_train, y_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), return_X_y=True)
X_train = np.array(X_train)

norm_vect_train = Cleaner.newsgroups(X_train, subset='train', verbose=True)

tuned_parameters = [{'n_estimators': [125, 175, 200, 225]}]

# 5-fold cross validation using an AdaBoost clf with fixed params
print('Cross-validating...')
clf = AdaBoostClassifier(learning_rate=0.8, random_state=0)
clf = GridSearchCV(clf, tuned_parameters, cv=5, refit=False, verbose=3)
clf.fit(norm_vect_train, y_train)
scores = clf.cv_results_['mean_test_score'].round(3)

print('scores:', scores)
