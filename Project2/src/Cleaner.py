import string
import numpy as np
import nltk
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

nltk.download('punkt')

# Update stopwords list with punctuation
sw = stopwords.words('english') + list(string.punctuation)

stemmer = SnowballStemmer('english')


def custom_tokenizer(doc):
    tokens = word_tokenize(doc)
    return [stemmer.stem(token) for token in tokens]


vectorizer = TfidfVectorizer(strip_accents='ascii', lowercase=True,
                             tokenizer=custom_tokenizer,
                             analyzer='word', stop_words=sw,
                             ngram_range=(1, 2))

normalizer = Normalizer()


class Cleaner:

    @staticmethod
    def newsgroups(X, subset, verbose=False):
        X_train = np.array(X)

        if verbose:
            print('Vectorizing {} data...'.format(subset))

        if subset == 'train':
            vect_train = vectorizer.fit_transform(X_train)
        elif subset == 'test':
            vect_train = vectorizer.transform(X_train)
        else:
            raise ValueError

        if verbose:
            print('Normalizing {} data...'.format(subset))

        norm_vect_train = normalizer.transform(vect_train)

        return norm_vect_train

    @staticmethod
    def IMDB(X, y, verbose=False):
        pass
