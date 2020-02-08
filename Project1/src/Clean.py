from Project1.src.Processor import Processor
class Clean:

    @staticmethod
    def adult(X):
        binaryCols = {
            "sex": {"Male": 0, "Female": 1},
            "salary": {">50K": 0, "<=50K": 1}
        }
        X = X.copy()

        X = Processor.removeMissing(X)
        X = Processor.toBinaryCol(X, binaryCols)

        X = Processor.normalize(X, ["fnlwgt", "hours-per-week"])
        Y = X["salary"]
        X = X.iloc[:, :-1]
        X = Processor.OHE(X)

        countryCols = ["native-country_Cambodia", "native-country_England", "native-country_Puerto-Rico",
                       "native-country_Canada", "native-country_Outlying-US(Guam-USVI-etc)", "native-country_India",
                       "native-country_Japan", "native-country_Greece", "native-country_South", "native-country_China",
                       "native-country_Cuba", "native-country_Iran", "native-country_Honduras", "native-country_Italy",
                       "native-country_Poland", "native-country_Jamaica", "native-country_Vietnam",
                       "native-country_Portugal", "native-country_Ireland", "native-country_France",
                       "native-country_Dominican-Republic", "native-country_Laos", "native-country_Ecuador",
                       "native-country_Taiwan", "native-country_Haiti", "native-country_Columbia",
                       "native-country_Hungary", "native-country_Guatemala", "native-country_Nicaragua",
                       "native-country_Scotland", "native-country_Thailand", "native-country_Yugoslavia",
                       "native-country_El-Salvador", "native-country_Trinadad&Tobago", "native-country_Peru",
                       "native-country_Hong", "native-country_Holand-Netherlands"]

        X = X.drop(columns=(["capital-gain", "capital-loss", "education-num"] + countryCols))

        # X['hours-per-week'].value_counts().plot(x='Age', linestyle="None", marker='o')

        # X['native-country'].hist(grid=False)

        # X.plot.scatter(x="relationship", y="age",  c='salary', colormap='autumn')
        # plt.waitforbuttonpress()

        return [X, Y]


    @staticmethod
    def Ionosphere(X):
        binaryCols = {
            "signal": {"g": 1, "b": 0}
        }
        X = X.copy()
        X = Processor.removeMissing(X)
        X = X.drop(columns=['col0', 'col1'])
        X = Processor.toBinaryCol(X, binaryCols)
        Y = X["signal"]
        X = X.iloc[:, :-1]

        return [X, Y]