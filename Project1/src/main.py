from Project1.src.LogisticRegression import LogisticRegression
from Project1.src.NaiveBayes import NaiveBayes
from Project1.src.Processor import Processor
from Project1.src.Clean import Clean
from Project1.src.CrossValidation import cross_validation
import numpy as np
import matplotlib.pyplot as plt

ds = "phishing"

if ds == "adult":
    path = "./datasets/adult/adult.data"
    header = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
              'relationship',
              'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'salary']

    All = Processor.read(path, header)

    [X, Y] = Clean.adult(All)

    print(X.shape)

    [X_train, X_test, Y_train, Y_test] = Processor.split(X, Y, train=0.8)

    model = LogisticRegression()

    print(cross_validation(5, X_train.to_numpy(), Processor.ToNumpyCol(Y_train), model, max_gradient=0.8, learning_rate=0.01))


elif ds == "ionosphere":
    path = "./datasets/ionosphere/ionosphere.data"

    header = ["{}{}".format("col", x) for x in range(33 + 1)]
    header.append("signal")

    All = Processor.read(path, header)

    [X, Y] = Clean.Ionosphere(All)

    [X_train, X_test, Y_train, Y_test] = Processor.split(X, Y, train=0.8)

    model = LogisticRegression()

    print(cross_validation(5, X_train.to_numpy(), Processor.ToNumpyCol(Y_train), model))

elif ds == "phishing":
    path = "./datasets/phishing/phishing.arff"
    header = ["ip", "url-length", "short-service", "at", "dslash", "prefix-suffix", "subdomain", "ssl", "domain-len",
              "favicon", "port", "https", "request-url", "url-anchor", "link-tags", "sfh", "email", "abnormal-url",
              "redirect", "mouseover", "right-click", "popup", "iframe", "domain-age", "dns", "web-traffic", "page-rank",
              "google", "links-to-page", "stats", "result"]
    all = Processor.read(path, header)

    [X, Y] = Clean.phishing(all)

    [X_train, X_test, Y_train, Y_test] = Processor.split(X, Y, train=0.8)

    model = LogisticRegression()

    print(cross_validation(5, X_train.to_numpy(), Processor.ToNumpyCol(Y_train), model))

elif ds == "ttt":
    header = ["tl", "tm", "tr", "ml", "mm", "mr", "bl", "bm", "br", "result"]

