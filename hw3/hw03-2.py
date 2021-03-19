# ID: 2018116323 (undergraduate)
# NAME: DaeHeon Yoon
# File name: hw03-2.py
# Platform: Python 3.8.8 on Windows 10
# Required Package(s): numpy pandas matplotlib sklearn

import warnings

import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

print(__doc__)

data = load_digits().data
target = load_digits().target

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)
X_train = X_train / 255.
X_test = X_test / 255.

X_train.shape, X_test.shape

mlp = MLPClassifier(hidden_layer_sizes=(30,), max_iter=500, alpha=1e-4,
                    solver='sgd', verbose=10, random_state=1,
                    learning_rate_init=.1)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning,
                            module="sklearn")
    mlp.fit(X_train, y_train)

print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))

fig, axes = plt.subplots(4, 4)
# use global min / max to ensure all weights are shown on the same scale
vmin, vmax = mlp.coefs_[0].min(), mlp.coefs_[0].max()
for coef, ax in zip(mlp.coefs_[0].T, axes.ravel()):
    ax.matshow(coef.reshape(8, 8), cmap=plt.cm.gray, vmin=.5 * vmin,
               vmax=.5 * vmax)
    ax.set_xticks(())
    ax.set_yticks(())

plt.show()
