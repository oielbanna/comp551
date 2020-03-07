"""
This script is used for the final evaluation of the AdaBoost model.
The hyper parameters used were chosen using the crossvalidation script: AdaBoostCV.py
"""
from sklearn.datasets import fetch_20newsgroups
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from Project2.src.Cleaner import Cleaner

# Get data and convert to numpy array when needed
print('Fetching data...')
X_train, y_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), return_X_y=True)
X_test, y_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), return_X_y=True)

norm_vect_train = Cleaner.newsgroups(X_train, subset='train', verbose=True)
norm_vect_test = Cleaner.newsgroups(X_test, subset='test', verbose=True)

# Instantiate model, train, and get predictions on test set
print('Training model...')
clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2), n_estimators=125, learning_rate=0.8)
clf.fit(norm_vect_train, y_train)

print('Predicting...')
y_hat = clf.predict(norm_vect_test)

# Evaluate the model
print('Accuracy score on the training set ' + str(clf.score(norm_vect_train, y_train)))
print('Accuracy score on the testing set ' + str(accuracy_score(y_test, y_hat)))
