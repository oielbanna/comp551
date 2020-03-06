from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
import string

import nltk
import nltk.tokenize as tokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

import pandas as pd

nltk.download('stopwords')

ps = PorterStemmer()

# token_pattern specifies what is considered a word. This  is basically saying any word with strictly text and length (3,10)
# min_df removes features (words) with count less than 5
# strip_accents removes some weird words that have accents
vec = TfidfVectorizer(stop_words=stopwords.words('english'),
                      strip_accents='ascii',
                      min_df=5,
                      # max_features=5000,
                      token_pattern=r"(?u)\b[a-zA-Z]{3,10}\b")


def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)


class Cleaner:
    @staticmethod
    def newsgroups(X, y, verbose=False):
        # print(X)
        # X  = [
        #     "about, Penny bought bright \n\t blue fishes.",
        #     "Penny bought a bright blue and orange fish.",
        #     "The fish \t\t\t !fished fishs'.",
        #     "I'm fishing omar@gmail.com--------------------- fish.",
        #     "I hate blue bugs",
        #     "A blue bug12 ate a fish",
        #     "fish 98"
        # ]

        cleaned_X = []
        punctuations = string.punctuation.replace("'", "")

        for sentence in X:
            #     remove emails
            #     sentence = re.sub('\S*[@>]\S*\s?','', sentence)
            #     remove punctuation
            sentence = sentence.translate(str.maketrans('', '', punctuations))

            # remove words that are less than len = 3 and do word stemming
            sentence = " ".join([ps.stem(word) for word in sentence.split(" ") if len(word) > 3])

            cleaned_X.append(sentence)

        if verbose:
            print('Vectorizing data...')

        matrix_x = vec.fit_transform(cleaned_X)

        results_x = pd.DataFrame(matrix_x.toarray(), columns=vec.get_feature_names())

        print(vec.get_feature_names())

        if verbose:
            print('Normalizing data...')
        normalizer = Normalizer()
        norm_vect_train = normalizer.fit_transform(results_x.to_numpy())

        return norm_vect_train, y

    @staticmethod
    def IMDB(X, y, verbose=False):
        pass
