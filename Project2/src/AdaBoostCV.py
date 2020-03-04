import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV

# Get data and convert to numpy array when needed
print('Fetching data...')
X_train, y_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), return_X_y=True)
X_train = np.array(X_train)

# Pre-process the text data by applying tf-idf vectorization and normalizing
print('Vectorizing data...')
vectorizer = TfidfVectorizer()
vect_train = vectorizer.fit_transform(X_train)

print('Normalizing data...')
normalizer = Normalizer().fit(X=vect_train)
norm_vect_train = normalizer.transform(vect_train)

# Uncomment parameters to tune
tuned_parameters = [{'learning_rate': [0.5, 0.7, 0.8, 0.9, 1.2, 1.5]}]

# 5-fold cross validation using an AdaBoost clf with fixed params
print('Cross-validating...')
clf = AdaBoostClassifier(n_estimators=50, random_state=0)
clf = GridSearchCV(clf, tuned_parameters, cv=5, refit=False)
clf.fit(norm_vect_train, y_train)
scores = clf.cv_results_['mean_test_score']

print('scores:', scores)
