
import matplotlib.pyplot as plt

import sklearn.datasets as datasets
from sklearn.preprocessing import Normalizer
import sklearn.metrics as metrics
from sklearn.pipeline import Pipeline

from sklearn.externals.six import StringIO
import pydotplus
from IPython.display import Image

from sklearn import tree
from sklearn.model_selection import train_test_split

newsgroups = datasets.fetch_20newsgroups_vectorized(subset='test', remove=('headers', 'footers', 'quotes'))
x = newsgroups.data
y = newsgroups.target
clf = tree.DecisionTreeClassifier(max_depth = 400)
x_train,x_test,y_train,y_test = train_test_split(x,y)

norm = Normalizer().fit(x_train, y_train)
x_train = norm.transform(x_train)

fig = clf.fit(x_train,y_train)
tree.plot_tree(fig)
# plt.show()

dot_data = StringIO()
tree.export_graphviz(fig, out_file=dot_data)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# Image(graph.create_png())
graph.write_png("i.png")
""""
# Get data and convert to numpy array when needed
print('Fetching data...')
newsgroups = datasets.fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), shuffle=True)
newsgroups_test = datasets.fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))


X_train = newsgroups.data
y_train = newsgroups.target

X_test = newsgroups_test.data
y_test = newsgroups_test.target

X_train = np.array(X_train)

# Pre-process the text data by applying tf-idf vectorization and normalizing
print('Vectorizing data...')
vectorizer = TfidfVectorizer()
vect_train = vectorizer.fit_transform(X_train)
vect_test = vectorizer.transform(X_test)

print('Normalizing data...')
normalizer = Normalizer().fit(X=vect_train)
norm_vect_train = normalizer.transform(vect_train)
norm_vect_test = normalizer.transform(vect_test)

# Instantiate model, train, and get predictions on test set
print('Training model...')
clf = DecisionTreeClassifier(criterion='gini', random_state=3, max_depth=300)

clf.fit(norm_vect_train, y_train)

print('Predicting...')
y_hat = clf.predict(norm_vect_test)

# Evaluate the model
print('Accuracy score on the training set ' + str(clf.score(norm_vect_train, y_train)))
print('Accuracy score on the testing set ' + str(accuracy_score(y_test, y_hat)))


"""