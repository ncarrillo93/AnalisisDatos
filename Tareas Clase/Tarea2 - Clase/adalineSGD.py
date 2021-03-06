import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
class AdalineSGD(object):

    def __init__(self, eta=0.01, n_iter=10,
        shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state

    def fit(self, X, y):
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        """Initialize weights to small random numbers"""
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01,
        size=1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        """Apply Adaline learning rule to update the weights"""
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]
    def activation(self, X):
        """Compute linear activation"""
        return X

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X))>= 0.0, 1, -1)

def plot_decision_regions(X, y, classifier, resolution=0.02) :
    #Setup market generator and colormap
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue','lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y)) ] )

    # plot the decision surface
    x1_min, x1_max = X[:,0].min() -1, X[:, 0].max() + 1
    x2_min, x2_max = X[:,1].min() -1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                        np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())

    #plot class example
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y== cl,0], y = X[y == cl, 1], alpha=0.8, c=colors[idx], marker=markers[idx], label=cl, edgecolor='black')

#Codigo Principal - Largo de sepalo, ancho de sepalo, largo de petalo, ancho de petalo, especie
df = pd.read_csv('Tarea2/iris.data', header=None, encoding='utf-8')
y   = df.iloc[0:100, 4].values
y   = np.where(y == 'Iris-setosa', -1, 1)

y_1 = df.iloc[0:100, 4].values
y_1 = np.where(y_1 == 'Iris-versicolor', 1, -1)

y_2 = df.iloc[0:100, 4].values
y_2 = np.where(y_2 == 'Iris-virginica', -1, 1)
#Extraer largo de sepalo y ancho de sepalo
X   = df.iloc[0:100, [0,1]].values
X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

X1   = df.iloc[0:100, [0,2]].values
X1_std = np.copy(X1)
X1_std[:,0] = (X1[:,0] - X1[:,0].mean()) / X1[:,0].std()
X1_std[:,1] = (X1[:,1] - X1[:,1].mean()) / X1[:,1].std()

X2   = df.iloc[0:100, [0,3]].values
X2_std = np.copy(X2)
X2_std[:,0] = (X2[:,0] - X2[:,0].mean()) / X2[:,0].std()
X2_std[:,1] = (X2[:,1] - X2[:,1].mean()) / X2[:,1].std()

X3   = df.iloc[0:100, [1,2]].values
X3_std = np.copy(X3)
X3_std[:,0] = (X3[:,0] - X3[:,0].mean()) / X3[:,0].std()
X3_std[:,1] = (X3[:,1] - X3[:,1].mean()) / X3[:,1].std()

X4   = df.iloc[0:100, [1,3]].values
X4_std = np.copy(X4)
X4_std[:,0] = (X4[:,0] - X4[:,0].mean()) / X4[:,0].std()
X4_std[:,1] = (X4[:,1] - X4[:,1].mean()) / X4[:,1].std()

X5   = df.iloc[0:100, [2,3]].values
X5_std = np.copy(X5)
X5_std[:,0] = (X5[:,0] - X5[:,0].mean()) / X5[:,0].std()
X5_std[:,1] = (X5[:,1] - X5[:,1].mean()) / X5[:,1].std()



ada_sgd = AdalineSGD(n_iter=10000, eta=0.0001, random_state=1)
ada_sgd.fit(X1_std, y_1)
plot_decision_regions(X1_std, y_1, classifier=ada_sgd)
plt.title('Adaline - Stochastic Gradient Descent')
plt.xlabel('Largo Sepalo [standardized]')
plt.ylabel('Largo Petalo [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
plt.plot(range(1, len(ada_sgd.cost_) + 1), ada_sgd.cost_,marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.tight_layout()
plt.show()
