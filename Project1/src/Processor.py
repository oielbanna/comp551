import numpy as np
import pandas as pd

class Processor:
    def __init__(self, filepath, header, headerTypes):
        if(len(header) != len(headerTypes)):
            raise Exception('Header array and headerTypes array should be the same length')
        self.path = filepath
        self.header = header
        self.types = headerTypes
        self.raw = self.__read(self.path, header)
        self.data = []

    # removing all rows with '?'
    def removeMissing(self):
        X = self.data
        if(len(self.data) == 0):
            X = self.raw

        for i, head in enumerate(self.header):
            if(self.types[i] == np.str):
                 X = X[~X[head].str.contains("\?")]

        self.data = X
        return self.data

    def fillMissing(self):
        pass

    def toBinaryCol(self, name):
        pass

    def analyze(self):
        pass

    def __read(self, path, header):
        return pd.read_csv(path, names=header, header=None)

    def __str__(self):
     return self.raw.head().to_string()