# import numpy as np
import pandas as pd

class Processor:
    # removing all rows with '?'
    @staticmethod
    def removeMissing(X):
        return X.dropna(axis=0)

    @staticmethod
    def removeOutliers(X):
        pass

    @staticmethod
    def fillMissing(X):
        vals = {}
        for i in X.columns:
            vals[i] = round(X[i].describe()['mean'])
        return X.fillna(value=vals)

    @staticmethod
    def toBinaryCol(X, dic):
        X.replace(dic, inplace=True)
        return X

    # One Hot Encoding for all columns with type object (ie categorical cols)
    @staticmethod
    def OHE(X, cols=[]):
        if len(cols) == 0:
            cols = list(X.select_dtypes(include=['object']).columns)
        return pd.get_dummies(X, columns=cols)

    @staticmethod
    def normalize(X, cols):
        X = X.copy()
        for c in cols:
            X[c] = (X[c] - X[c].mean()) / (X[c].max() - X[c].min())
        return X

    @staticmethod
    def analyze(X):
        return X.describe()

    @staticmethod
    def split(X, Y, train=0.8):
        R, C = X.shape
        n = int(R * train)
        X_train = X.iloc[:n]
        X_test = X.iloc[n:]
        Y_train = Y.iloc[:n]
        Y_test = Y.iloc[n:]
        return [X_train, X_test, Y_train, Y_test]

    @staticmethod
    def read(path, header):
        if header is None or len(header) == 0:
            return pd.read_csv(path, na_values="?", skipinitialspace=True)
        return pd.read_csv(path,
                           header=None, names=header, na_values="?", skipinitialspace=True)

    @staticmethod
    def ToNumpyCol(X):
        X = X.to_numpy()
        return X.reshape((X.shape[0], 1))
