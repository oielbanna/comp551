
import matplotlib.pyplot as plt
from Project2.src.Cleaner import Cleaner
import sklearn.datasets as datasets
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.metrics as metrics
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# newsgroups_train = datasets.fetch_20newsgroups_vectorized(subset='train', remove=('headers', 'footers', 'quotes'), shuffle=True)
# x = newsgroups_train.data
# y = newsgroups_train.target
#
# x_train,x_test,y_train,y_test = train_test_split(x,y)
#
# clf = tree.DecisionTreeClassifier(max_depth=400)
# norm = Normalizer().fit(x_train, y_train)
# x_train = norm.transform(x_train)
#
# fig = clf.fit(x_train,y_train)
#
# y_hat = clf.predict(x_test)
#
# print('Accuracy score on the training set ' + str(clf.score(x_train, y_train)))
# print('Accuracy score on the testing set ' + str(metrics.accuracy_score(y_test, y_hat)))
# tree.plot_tree(fig)
# plt.show()

""""
# Get data and convert to numpy array when needed
print('Fetching data...')
newsgroups = datasets.fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), shuffle=True)
newsgroups_test = datasets.fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))


X_train = newsgroups.data
y_train = newsgroups.target

X_test = newsgroups_test.data
y_test = newsgroups_test.target

X_train = np.array(X_train)

# Pre-process the text data by applying tf-idf vectorization and normalizing
print('Vectorizing data...')
vectorizer = TfidfVectorizer()
vect_train = vectorizer.fit_transform(X_train)
vect_test = vectorizer.transform(X_test)

print('Normalizing data...')
normalizer = Normalizer().fit(X=vect_train)
norm_vect_train = normalizer.transform(vect_train)
norm_vect_test = normalizer.transform(vect_test)

# Instantiate model, train, and get predictions on test set
print('Training model...')
clf = DecisionTreeClassifier(criterion='gini', random_state=3, max_depth=300)

clf.fit(norm_vect_train, y_train)

print('Predicting...')
y_hat = clf.predict(norm_vect_test)

# Evaluate the model
print('Accuracy score on the training set ' + str(clf.score(norm_vect_train, y_train)))
print('Accuracy score on the testing set ' + str(accuracy_score(y_test, y_hat)))

"""


newsgroups_train = datasets.fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), shuffle=False)

x = newsgroups_train.data
y = newsgroups_train.target

x_train,x_test,y_train,y_test = train_test_split(x,y)

x_train = Cleaner.newsgroups(x_train, subset='train', verbose=True)
x_test = Cleaner.newsgroups(x_test, subset='test', verbose=True)

print(x_train.shape, x_test.shape)

print('Training model...')
clf = tree.DecisionTreeClassifier(criterion="gini",
                                  random_state=0,)
clf.fit(x_train, y_train)

print(clf.get_depth())

print('Predicting...')
y_hat = clf.predict(x_test)

# Evaluate the model
print('Accuracy score on the training set ' + str(clf.score(x_train, y_train)))
print('Accuracy score on the testing set ' + str(accuracy_score(y_test, y_hat)))



# tuned_parameters = [{'max_depth': [100, 150, 200, 250, 300, 350]}]
#
# # 5-fold cross validation using an AdaBoost clf with fixed params
# print('Cross-validating...')
# clf = tree.DecisionTreeClassifier(criterion='gini')
# clf = GridSearchCV(clf, tuned_parameters, cv=5, refit=False, verbose=3)
# clf.fit(x, y)
# scores = clf.cv_results_['mean_test_score'].round(3)
#
# print('scores:', scores)
