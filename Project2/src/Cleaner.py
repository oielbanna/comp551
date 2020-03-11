import string
import numpy as np
import nltk
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re

# nltk.download('punkt')

# Update stopwords list with punctuation
sw = stopwords.words('english') + list(string.punctuation) + ['arent', 'couldnt', 'didnt', 'doesnt', 'dont', 'hadnt', 'hasnt', 'havent', 'isnt', 'mightnt', 'mustnt', 'neednt', 'shant', 'shes', 'shouldnt', 'shouldv', 'thatll', 'wasnt', 'werent', 'wont', 'wouldnt', 'youd', 'youll', 'youv', '``', 'abov', 'ani', 'becaus', 'befor', 'doe', 'dure', 'onc', 'onli', 'ourselv', 'themselv', 'veri', 'whi', 'yourselv']

stemmer = SnowballStemmer('english')


def custom_tokenizer(doc):
    doc = re.sub(r"[@\-/\\#&()*{`}±~'<?^>$.:\[\]|;_,=]+", '', doc)
    tokens = word_tokenize(doc.strip())
    stemmed = [stemmer.stem(token) for token in tokens]
    # print(stemmed)
    return stemmed


vectorizer = TfidfVectorizer(strip_accents='ascii', lowercase=True,
                             tokenizer=custom_tokenizer,
                             analyzer='word', stop_words=sw,
                             ngram_range=(1, 3),
                             max_features=20000,
                             max_df=0.95)

normalizer = Normalizer()


class Cleaner:

    @staticmethod
    def clean(X, subset, verbose=False):
        X_train = np.array(X)

        if verbose:
            print('\tVectorizing {} data...'.format(subset))

        if subset == 'train':
            vect_train = vectorizer.fit_transform(X_train)
        elif subset == 'test':
            vect_train = vectorizer.transform(X_train)
        else:
            raise ValueError

        # print(vectorizer.get_feature_names())
        if verbose:
            print('\tNormalizing {} data...'.format(subset))

        norm_vect_train = normalizer.transform(vect_train)

        return norm_vect_train
