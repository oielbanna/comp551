from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.pipeline import Pipeline
import nltk
import numpy as np

#negative is mapped to a target(label) index of 1, positive to 0
twenty_train = fetch_20newsgroups(subset='train', shuffle=True, remove=('headers', 'footers', 'quotes'), random_state=42)
"""twenty_test = fetch_20newsgroups(subset='test', shuffle=True, remove=('headers', 'footers', 'quotes'), random_state=42)"""

x_train, x_test, y_train, y_test = train_test_split(twenty_train.data, twenty_train.target, test_size = 0.20, random_state=42)

"""x_train = twenty_train.data
x_test = twenty_test.data
y_train = twenty_train.target
y_test = twenty_test.target"""

#Regular SVM with linear kernel and no preprocessing, Parameters: gamma = 0.001, C = 100.
SVM_clf = Pipeline([
    ('count_vector', CountVectorizer(tokenizer=nltk.word_tokenize,encoding='latin-1')),
    ('tf_transformer', TfidfTransformer()),
    ('clf', svm.LinearSVC()),
])

#Regular linear regression model
LR_clf = Pipeline([
    ('count_vector', CountVectorizer(tokenizer=nltk.word_tokenize,encoding='latin-1')),
    ('tf_transformer', TfidfTransformer()),
    ('clf', LogisticRegression(max_iter=1000)),
])

SVM_clf.fit(x_train, y_train)
y_pred = SVM_clf.predict(x_test)
print("~~~~~~ SVM ~~~~~~")
print("SVM Accuracy on training set: ", SVM_clf.score(x_train, y_train))
print("SVM Accuracy on test set: ", metrics.accuracy_score(y_test, y_pred))
"""print("SVM Precision: ", metrics.precision_score(y_test, y_pred))
print("SVM Recall: ", metrics.recall_score(y_test, y_pred))
print("SVM F1: ", metrics.f1_score(y_test, y_pred))"""

LR_clf.fit(x_train, y_train)
y_pred = LR_clf.predict(x_test)
print("~~~~~~ Logistic Regression ~~~~~~")
print("LR Accuracy on training set: ", LR_clf.score(x_train, y_train))
print("LR Accuracy: ", metrics.accuracy_score(y_test, y_pred))
"""print("LR Precision: ", metrics.precision_score(y_test, y_pred))
print("LR Recall: ", metrics.recall_score(y_test, y_pred))
print("LR F1: ", metrics.f1_score(y_test, y_pred))"""
