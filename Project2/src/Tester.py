from sklearn.datasets import fetch_20newsgroups
import pandas as pd
from Project2.src.Cleaner import Cleaner
from Project2.src.Models import Models

from sklearn import ensemble, tree, neighbors, linear_model, svm, metrics


def getTrainTestDataset(dataset):
    if dataset == "IMDB":
        train_path = "../datasets/train_reviews.csv"
        train_imdb = pd.read_csv(train_path, skipinitialspace=True)
        X_train = train_imdb['reviews']
        y_train = train_imdb['target']

        test_path = "../datasets/test_reviews.csv"
        test_imdb = pd.read_csv(test_path, skipinitialspace=True)
        X_test = test_imdb['reviews']
        y_test = test_imdb['target']
    else:
        X_train, y_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'),
                                              return_X_y=True)
        X_test, y_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), return_X_y=True)

    return [X_train, X_test, y_train, y_test]


def getModel(dataset, model):
    if model == Models.RandomForest:
        if dataset == "IMDB":
            return ensemble.RandomForestClassifier(
                criterion='gini',
                max_depth=600,
                max_features=0.8,
                max_leaf_nodes=100,
                min_impurity_decrease=0.0001,
                n_estimators=50,
                oob_score=True
            )
        else:
            return ensemble.RandomForestClassifier(
                criterion='gini',
                max_depth=450,
                max_features=None,
                min_impurity_decrease=0,
                n_estimators=300,
                oob_score=True,
                random_state=30
            )

    elif model == Models.DecisionTree:
        if dataset == "IMDB":
            return tree.DecisionTreeClassifier(
                max_depth=450,
                min_impurity_decrease=0.0001,
                max_leaf_nodes=600,
                random_state=30,
                max_features=0.4,
                criterion='gini',
                splitter="best",
                ccp_alpha=0.00055
            )
        else:
            return tree.DecisionTreeClassifier(
                max_depth=600,
                min_impurity_decrease=0.0001,
                max_leaf_nodes=100,
                max_features=0.8,
                splitter="random",
                ccp_alpha=0.00025,
                criterion='gini',
            )
    elif model == Models.AdaBoost:
        if dataset == "IMDB":
            return ensemble.AdaBoostClassifier()
        else:
            return ensemble.AdaBoostClassifier()

    elif model == Models.KNN:
        if dataset == "IMDB":
            return neighbors.KNeighborsClassifier()
        else:
            return neighbors.KNeighborsClassifier()

    elif model == Models.LogisticRegression:
        if dataset == "IMDB":
            return linear_model.LogisticRegression(C=1.0, dual=False, max_iter=1000, penalty='l1', solver='liblinear', tol=0.1)
        else:
            return linear_model.LogisticRegression(C=1.0, dual=False, max_iter=100, penalty='l2', solver='saga', tol=0.01)

    elif model == Models.SVM:
        if dataset == "IMDB":
            return svm.LinearSVC(C=0.1, dual=False, loss='squared_hinge', max_iter=1000, penalty='l2', tol=0.1)
        else:
            return svm.LinearSVC(C=1.0, dual=True, fit_intercept=True, loss='squared_hinge', max_iter=5000, penalty='l2', tol=0.01)


class Tester:
    @staticmethod
    def run(dataset, model, verbose=False):
        [X_train, X_test, y_train, y_test] = getTrainTestDataset(dataset)

        if verbose:
            print("Cleaning train data...")
        X_train = Cleaner.clean(X_train, 'train', verbose)
        if verbose:
            print("Cleaning test data...")
        X_test = Cleaner.clean(X_test, 'test', verbose)

        clf = getModel(dataset, model)

        if verbose:
            print("Fitting model for {}...".format(model))
        clf.fit(X_train, y_train)

        if verbose:
            print("Predicting...")
        y_hat = clf.predict(X_test)

        return [round(clf.score(X_train, y_train), 3), round(metrics.accuracy_score(y_test, y_hat), 3)]