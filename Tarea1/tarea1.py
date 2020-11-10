import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

class Perceptron(object):

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
df = pd.read_csv('iris.data', header=None, encoding='utf-8')
df.tail()
df.head()

# select setosa and versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
# extract sepal length and petal length
X = df.iloc[0:100, [0, 2]].values


# plot data sepal length vs petal length (sl,pl) [5.1 1.4] 1era coord
plt.subplot(231)
plt.scatter(X[:50, 0], X[:50, 1],color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],color='blue', marker='x', label='versicolor')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
#plt.show()

# select setosa and versicolor
#sw_1 = df.iloc[0:100, 1].values
sw_1 = df.iloc[0:100, 4].values
sw_1 = np.where(sw_1 == 'Iris-setosa', -1, 1)
# extract sepal length and sepal width
sl_1 = df.iloc[0:100, [0, 1]].values
#plot data sepal length vs sepal width [5.1 3.5]
plt.subplot(232)
plt.scatter(sl_1[:50, 0], sl_1[:50, 1],color='red', marker='o', label='setosa')
plt.scatter(sl_1[50:100, 0], sl_1[50:100, 1],color='blue', marker='x', label='versicolor')
plt.xlabel('sepal length [cm]')
plt.ylabel('sepal width [cm]')
plt.legend(loc='upper left')
#plt.show()

# select setosa and versicolor
#pw_1 = df.iloc[0:100, 1].values
pw_1 = df.iloc[0:100, 4].values
pw_1 = np.where(pw_1 == 'Iris-setosa', -1, 1)
# extract sepal length and petal width
sl_2 = df.iloc[0:100, [0, 3]].values
#plot data sepal length vs petal width [5.1 0.2]
plt.subplot(233)
plt.scatter(sl_2[:50, 0], sl_2[:50, 1],color='red', marker='o', label='setosa')
plt.scatter(sl_2[50:100, 0], sl_2[50:100, 1],color='blue', marker='x', label='versicolor')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
#plt.show()

# select setosa and versicolor
#sw_2 = df.iloc[0:100, 2].values
sw_2 = df.iloc[0:100, 4].values
sw_2 = np.where(sw_2 == 'Iris-setosa', -1, 1)
# extract sepal width and petal length
pl = df.iloc[0:100, [1, 2]].values 
#plot data sepal width vs petal length [3.5 1.4]
plt.subplot(234)
plt.scatter(pl[:50, 0], pl[:50, 1],color='red', marker='o', label='setosa')
plt.scatter(pl[50:100, 0], pl[50:100, 1],color='blue', marker='x', label='versicolor')
plt.xlabel('sepal width [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
#plt.show()

# select setosa and versicolor
sw_3 = df.iloc[0:100, 4].values
sw_3 = np.where(sw_3 == 'Iris-setosa', -1, 1)
# extract sepal width and petal width
pw = df.iloc[0:100, [1, 3]].values 
#plot data sepal width vs petal width [3.5 0.2]
plt.subplot(235)
plt.scatter(pw[:50, 0], pw[:50, 1],color='red', marker='o', label='setosa')
plt.scatter(pw[50:100, 0], pw[50:100, 1],color='blue', marker='x', label='versicolor')
plt.xlabel('sepal width [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
#plt.show()

# select setosa and versicolor
pl_2 = df.iloc[0:100, 4].values
pl_2 = np.where(pl_2 == 'Iris-setosa', -1, 1)
# extract sepal width and petal width
pw_2 = df.iloc[0:100, [2, 3]].values 
print(pw_2[0])
#plot data petal length vs petal width [1.4 0.2]
plt.subplot(236)
plt.scatter(pw_2[:50, 0], pw_2[:50, 1],color='red', marker='o', label='setosa')
plt.scatter(pw_2[50:100, 0], pw_2[50:100, 1],color='blue', marker='x', label='versicolor')
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.savefig('img/otros_pares.png')
#plt.show()





#Perceptron
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
plt.subplot(231)
plt.plot(range(1, len(ppn.errors_) + 1),ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()

ppn1 = Perceptron(eta=0.1, n_iter=10)
ppn1.fit(sl_1, sw_1)
plt.subplot(232)
plt.plot(range(1, len(ppn1.errors_) + 1),ppn1.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')

ppn2 = Perceptron(eta=0.1, n_iter=10)
ppn2.fit(sl_2, pw_1)
plt.subplot(233)
plt.plot(range(1, len(ppn2.errors_) + 1),ppn2.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')

ppn3 = Perceptron(eta=0.1, n_iter=10)
ppn3.fit(pl, sw_2)
plt.subplot(234)
plt.plot(range(1, len(ppn3.errors_) + 1),ppn3.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')

ppn4 = Perceptron(eta=0.1, n_iter=10)
ppn4.fit(pw, sw_3)
plt.subplot(235)
plt.plot(range(1, len(ppn4.errors_) + 1),ppn4.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')

ppn5 = Perceptron(eta=0.1, n_iter=10)
ppn5.fit(pw_2, pl_2)
plt.subplot(236)
plt.plot(range(1, len(ppn5.errors_) + 1),ppn5.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')

plt.savefig('img/perceptron_otros_pares.png')
plt.show()

