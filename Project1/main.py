from src.Processor import Processor
import numpy as np
import pandas as pd

adult = "./datasets/adult/adult.data"
aheader = ['age','workclass','fnlwgt','education', 'education-num', 'marital-status','occupation','relationship','race', 'sex','capital-gain','capital-loss','hours-per-week','native-country', 'salary']
atypes = [np.int64, np.str, np.uint64, np.str, np.uint64, np.str, np.str, np.str, np.str, np.str, np.uint64, np.uint64, np.uint64, np.str, np.str]
aReader = Processor(adult, aheader, atypes)

# remove last col
# X = aReader.raw.iloc[:, :-1] 
X = aReader.raw
# for i in aheader:
#     print(i)
#     X = X[X[i] != '?']
# for i, row in X.iterrows():
#     for j, col in row.iteritems():
#         # print(type(col))
#         if(type(col) is str and "?" in col):
#             # print(row)
#             print(i)
#             # X.drop(axis=0, index=i)
#             break

aReader.removeMissing()
print(aReader.data)
# dt = np.dtype({'names':aheader, 'formats':atypes})
# X = np.loadtxt(adult,delimiter=", ", dtype=dt)

# df = pd.DataFrame(X)
# print(df.dtypes)