# ID: 2018116323 (undergraduate)
# NAME: DaeHeon Yoon
# File name: hw02-3.py
# Platform: Python 3.8.8 on Windows 10
# Required Package(s): numpy pandas matplotlib sklearn

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

class Perceptron:
    """
    Perceptron neuron
    """

    def __init__(self, learning_rate=0.1):
        """
        instantiate a new Perceptron

        :param learning_rate: coefficient used to tune the model
        response to training data
        """
        self.learning_rate = learning_rate
        self._b = 0.0  # y-intercept
        self._w = None  # weights assigned to input features
        # count of errors during each iteration
        self.misclassified_samples = []

    def fit(self, x: np.array, y: np.array, n_iter=10):
        """
        fit the Perceptron model on the training data

        :param x: samples to fit the model on
        :param y: labels of the training samples
        :param n_iter: number of training iterations 
        """
        self._b = 0.0
        self._w = np.zeros(x.shape[1])
        self.misclassified_samples = []

        for _ in range(n_iter):
            # counter of the errors during this training iteration
            errors = 0
            for xi, yi in zip(x, y):
                # for each sample compute the update value
                update = self.learning_rate * (yi - self.predict(xi))
                # and apply it to the y-intercept and weights array
                self._b += update
                self._w += update * xi
                errors += int(update != 0.0)

            self.misclassified_samples.append(errors)

    def f(self, x: np.array) -> float:
        """
        compute the output of the neuron
        :param x: input features
        :return: the output of the neuron
        """
        return np.dot(x, self._w) + self._b

    def predict(self, x: np.array):
        """
        convert the output of the neuron to a binary output
        :param x: input features
        :return: 1 if the output for the sample is positive (or zero),
        -1 otherwise
        """
        return np.where(self.f(x) >= 0, 1, -1)



df = pd.read_csv("wine.csv",sep=",",engine='python')
df.head()

x = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.set_title("Wine data set")
ax.set_xlabel("Alcohol")
ax.set_ylabel("Malic acid")
ax.set_zlabel("Ash")

# plot the samples
ax.scatter(x[:59, 0], x[:59, 1], x[:59, 2], color='red', 
           marker='o', s=4, edgecolor='red', label="1")
ax.scatter(x[59:130, 0], x[59:130, 1], x[59:130, 2], color='blue', 
           marker='^', s=4, edgecolor='blue', label="2")
ax.scatter(x[130:, 0], x[130:, 1], x[130:, 2], color='green', 
           marker='x', s=4, edgecolor='green', label="3")

plt.legend(loc='upper left')
plt.show()

# 1 vs 2
x = np.concatenate([x[0:59], x[59:130]])
y = np.concatenate([y[0:59], y[59:130]])

# map the labels to a binary integer value
y = np.where(y == 1, 1, -1)

# standardization of the input features
plt.hist(x[:, 0], bins=100)
plt.title("Features before standardization")
plt.savefig("./before.png", dpi=300)
plt.show()

for i in range(13):
    x[:, i] = (x[:, i] - x[:, i].mean()) / x[:, i].std()

plt.hist(x[:, 0], bins=100)
plt.title("Features after standardization")
plt.show()

# split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25,
                                                    random_state=0)

# train the model
classifier = Perceptron(learning_rate=0.01)
classifier.fit(x_train, y_train)

# plot the number of errors during each iteration
plt.plot(range(1, len(classifier.misclassified_samples) + 1),
         classifier.misclassified_samples, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Errors')
plt.show()

