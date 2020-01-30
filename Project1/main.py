from src.Processor import Processor

adult = "./datasets/adult/adult.data"
aheader = ['age','workclass','fnlwgt','education', 'education-num', 'marital-status','occupation','relationship','race', 'sex','capital-gain','capital-loss','hours-per-week','native-country', 'salary']

aReader = Processor(adult, aheader)


# remove last col
X = aReader.raw.iloc[:, :-1] 
# print(X['native-country']

# print(X.drop("?", axis=0).shape)