import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer


class Cleaner:
    @staticmethod
    def newsgroups(X, y, verbose=False):
        X_train = np.array(X)

        # Pre-process the text data by applying tf-idf vectorization and normalizing
        if verbose:
            print('Vectorizing data...')q
        vectorizer = TfidfVectorizer()
        vect_train = vectorizer.fit_transform(X_train)

        if verbose:
            print('Normalizing data...')
        normalizer = Normalizer()
        norm_vect_train = normalizer.fit_transform(vect_train)

        return norm_vect_train, y

    @staticmethod
    def IMDB(X, y, verbose=False):
        pass