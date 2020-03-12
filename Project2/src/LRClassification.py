"""
This script is used for the final evaluation of the KNN model.
The hyper parameters used were chosen using the crossvalidation script: KNNCV.py
"""
from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import LogisticRegression
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
clf = LogisticRegression(C=1.0, dual=False, max_iter=100, penalty='l2', solver='lbfgs', tol= 0.01)
clf.fit(norm_vect_train, y_train)

print('Predicting...')
y_hat = clf.predict(norm_vect_test)

# Evaluate the model
print('Accuracy score on the training set ' + str(round(clf.score(norm_vect_train, y_train), 3)))
print('Accuracy score on the testing set ' + str(round(accuracy_score(y_test, y_hat), 3)))