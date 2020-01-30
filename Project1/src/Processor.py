import numpy as np
import pandas as pd

class Processor:
    def __init__(self, filepath, header):
        self.path = filepath
        self.raw = self.__read(self.path, header)

    def clean(self):
        pass

    def analyze(self):
        pass

    def __read(self, path, header):
        return pd.read_csv(path, names=header, header=None)

    def __str__(self):
     return self.raw.head().to_string()