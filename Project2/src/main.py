
from Project2.src.Tester import Tester
from Project2.src.Models import Models

# dataset = "IMDB"
dataset = "20_newsgroups"

# model = Models.DecisionTree
model = Models.RandomForest
# model = Models.KNN
# model = Models.AdaBoost
# model = Models.LogisticRegression
# model = Models.SVM

[train, test] = Tester.run(dataset, model, verbose=True)
print("{} performance on {}".format(model, dataset))
print("\tTraining accuracy {}".format(train))
print("\tTesting accuracy {}".format(test))
