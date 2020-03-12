import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import string
import numpy as np





# Get data and convert to numpy array when needed
print('Fetching data...')
X_train, y_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), return_X_y=True)
X_train = np.array(X_train)

norm_vect_train = Cleaner.clean(X_train, subset='train', verbose=True)

tuned_parameters = [{'n_neighbors': [25, 50, 100, 150],
                     'weights': ['uniform', 'distance'],
                     'p': [1, 2]}]

# 5-fold cross validation using an AdaBoost clf with fixed params
print('Cross-validating...')
clf = KNeighborsClassifier()
clf = GridSearchCV(clf, tuned_parameters, cv=5, refit=False, verbose=3)
clf.fit(norm_vect_train, y_train)
scores = clf.cv_results_['mean_test_score'].round(3)

print('scores:', scores)