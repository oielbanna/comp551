# import numpy as np
import pandas as pd

class Processor:
    # removing all rows with '?'
    @staticmethod
    def removeMissing(X):
        # for i, head in enumerate(self.header):
        #     if(self.types[i] == str):
        #          X = X[~X[head].str.contains("\?")]
        return X.dropna(axis=0)

    @staticmethod
    def fillMissing(X):
        raise Exception("Method unimplemented.")
        obj_df = obj_df.fillna({"num_doors": "four"})

    @staticmethod
    def toBinaryCol(X, dic):
        X.replace(dic, inplace=True)
        return X

    # One Hot Encoding for all columns with type object (ie categorical cols)
    @staticmethod
    def OHE(X):
        cols = list(X.select_dtypes(include=['object']).columns)
        return pd.get_dummies(X, columns=cols)

    @staticmethod
    def analyze(X):
        return X.describe()
    
    @staticmethod
    def read(path, header):
        return pd.read_csv(path,
                  header=None, names=header, na_values="?",skipinitialspace=True)
