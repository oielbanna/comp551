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

tuned_parameters = [{'n_estimators': [50, 90, 120],
                     "max_depth": [200, 300, 400],
                     "max_leaf_nodes": [200, 300, 500, 600],
                     "max_features": [0.4, 0.8, None]}]

# 5-fold cross validation using a DecisionTree clf with fixed params
print('Cross-validating...')
clf = ensemble.RandomForestClassifier(
    min_impurity_decrease=0.0001,
    random_state=30,
    criterion='gini',
    ccp_alpha=0.0002
)
clf = GridSearchCV(clf, tuned_parameters, cv=5, refit=True, verbose=3, n_jobs=-1)
clf.fit(x, y)
scores = clf.cv_results_['mean_test_score'].round(3)
print("The best parameters are %s with a score of %0.3f"
      % (clf.best_params_, clf.best_score_))

print('scores:', scores)
