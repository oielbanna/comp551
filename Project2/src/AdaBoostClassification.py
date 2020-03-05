"""
This script is used for the final evaluation of the AdaBoost model.
The hyper parameters used were chosen using the crossvalidation script: AdaBoostCV.py
"""
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

# Get data and convert to numpy array when needed
print('Fetching data...')
X_train, y_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), return_X_y=True)
X_test, y_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), return_X_y=True)

X_train = np.array(X_train)
X_test = np.array(X_test)


# Pre-process the text data by applying tf-idf vectorization and normalizing
print('Vectorizing data...')
vectorizer = TfidfVectorizer()
vect_train = vectorizer.fit_transform(X_train)
vect_test = vectorizer.transform(X_test)

print(vect_train)

print('Normalizing data...')
normalizer = Normalizer().fit(X=vect_train)
norm_vect_train = normalizer.transform(vect_train)
norm_vect_test = normalizer.transform(vect_test)

# Instantiate model, train, and get predictions on test set
print('Training model...')
clf = AdaBoostClassifier(n_estimators=50, random_state=0)
clf.fit(norm_vect_train, y_train)

print('Predicting...')
y_hat = clf.predict(norm_vect_test)

# Evaluate the model
print('Accuracy score on the training set ' + str(clf.score(norm_vect_train, y_train)))
print('Accuracy score on the testing set ' + str(accuracy_score(y_test, y_hat)))
