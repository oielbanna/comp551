from Project3.src.mlp import MLP
import tensorflow as tf
from tensorflow.keras import datasets

import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
#
classes = ['plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
#

train_images = train_images[0: 100]
train_labels = train_labels[0: 100]
print("hello", len(classes))
nn = MLP([3072, 50, 30, 10], 'sigmoid')
nn.train(train_images, train_labels)
# nn.predict(test_images[0])