from Project1.src.Processor import Processor

adult = "./datasets/adult/adult.data"
aheader = ['age','workclass','fnlwgt','education', 'education-num', 'marital-status','occupation','relationship','race', 'sex','capital-gain','capital-loss','hours-per-week','native-country', 'salary']
adultBinaryCols = {
    "sex": {"Male": 0, "Female": 1},
    "salary": {">50K":0, "<=50K": 1}
}

"""
    **************************************
    SHOWING USAGE OF PROCESSOR CLASS BELOW
"""
# remove last col
# X = aReader.raw.iloc[:, :-1] 
X = Processor.read(adult, aheader)
X = Processor.removeMissing(X)
X = Processor.toBinaryCol(X, adultBinaryCols)
X = Processor.OHE(X)
print(X)
# print(*list(X.columns), sep = ", ")


# dt = np.dtype({'names':aheader, 'formats':atypes})
# X = np.loadtxt(adult,delimiter=", ", dtype=dt)
# df = pd.DataFrame(X)
# print(df.dtypes)