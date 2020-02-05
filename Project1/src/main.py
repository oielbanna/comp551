from Project1.src.LogisticRegression import LogisticRegression
from Project1.src.NaiveBayes import NaiveBayes
from Project1.src.Processor import Processor

adult = "../datasets/adult/adult.data"
aheader = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',
           'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'salary']
adultBinaryCols = {
    "sex": {"Male": 0, "Female": 1},
    "salary": {">50K": 0, "<=50K": 1}
}

"""
    **************************************
    SHOWING USAGE OF PROCESSOR CLASS BELOW
"""
# remove last col
X = Processor.read(adult, aheader)
X = Processor.removeMissing(X)
X = Processor.toBinaryCol(X, adultBinaryCols)
Y = X["salary"]
X = X.iloc[:, :-1]
X = Processor.OHE(X)

YHead = Y.head(25).to_numpy()
YHead = YHead.reshape((YHead.shape[0], 1))

model = NaiveBayes()
w = model.fit(X.head(25).to_numpy(), YHead)

#model = LogisticRegression()
#w = model.fit(X.head(25).to_numpy(), YHead)

X_test = X.tail(5)
Y_test = Y.tail(5)

print(model.predict(X_test.to_numpy()))
print(Y_test)
