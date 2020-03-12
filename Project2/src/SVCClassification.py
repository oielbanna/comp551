import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from Project2.src.Cleaner import Cleaner

# Get data and convert to numpy array when needed
print('Fetching data...')
path = "../datasets/train_reviews.csv"
newsgrous_train = pd.read_csv(path, skipinitialspace=True)

X_train = newsgrous_train['reviews']
y_train = newsgrous_train['target']

norm_vect_train = Cleaner.newsgroups(X_train, subset='train', verbose=True)
norm_vect_test = Cleaner.newsgroups(X_test, subset='test', verbose=True)

# Instantiate model, train, and get predictions on test set
print('Training model...')
clf = LinearSVC(C=0.01, max_iter=50000)
clf.fit(norm_vect_train, y_train)

print('Predicting...')
y_hat = clf.predict(norm_vect_test)

# Evaluate the model
print('Accuracy score on the training set ' + str(round(clf.score(norm_vect_train, y_train), 3)))
print('Accuracy score on the testing set ' + str(round(accuracy_score(y_test, y_hat), 3)))