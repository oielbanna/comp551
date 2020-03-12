from Project2.src.Cleaner import Cleaner
import sklearn.datasets as datasets

from sklearn import ensemble
from sklearn.model_selection import GridSearchCV

newsgroups_train = datasets.fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), shuffle=False)

x = newsgroups_train.data
y = newsgroups_train.target
#
# x_train,x_test,y_train,y_test = train_test_split(x,y)
#
# x_train = Cleaner.newsgroups(x_train, subset='train', verbose=True)
# x_test = Cleaner.newsgroups(x_test, subset='test', verbose=True)
#
# print(x_train.shape, x_test.shape)
#
# print('Training model...')
# clf = tree.DecisionTreeClassifier(criterion="gini",
#                                   random_state=0,)
# clf.fit(x_train, y_train)
#
# print(clf.get_depth())
#
# print('Predicting...')
# y_hat = clf.predict(x_test)
#
# # Evaluate the model
# print('Accuracy score on the training set ' + str(clf.score(x_train, y_train)))
# print('Accuracy score on the testing set ' + str(accuracy_score(y_test, y_hat)))


x = Cleaner.clean(x, subset='train', verbose=True)

# tuned_parameters = [{'n_estimators': [100, 150, 200, 250]}]
tuned_parameters = [{'warm_start': [True, False]}]

# 5-fold cross validation using a DecisionTree clf with fixed params
print('Cross-validating...')
clf = ensemble.RandomForestClassifier(
    criterion='gini',
    max_depth=450,
    min_impurity_decrease=0,
    max_leaf_nodes=600,
    random_state=30,
    n_estimators=200,
    # warm_start=True,
    oob_score=True
)
clf = GridSearchCV(clf, tuned_parameters, cv=5, refit=False, verbose=3)
clf.fit(x, y)
scores = clf.cv_results_['mean_test_score'].round(3)

print('scores:', scores)
