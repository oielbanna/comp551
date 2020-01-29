import numpy as np
import pandas as pd

class Processor:
    def __init__(self, filepath):
        self.path = filepath
        self.raw = self.__read(self.path)

    def clean(self):
        pass

    def analyze(self):
        pass

    def __read(self, path):
        return pd.read_csv(path)

    def __str__(self):
     return self.raw.head().to_string()