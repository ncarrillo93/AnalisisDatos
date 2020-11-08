
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

#Clase perceptron
class perceptron(object):
    
    def __init__(self,eta=0.01,n_iter=50,random_state=1):
        self.eta=eta
        self.n_iter= n_iter
        self.random_state=random_state
    
    def fit(self, x, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_=rgen.normal(loc=0.0, scale= 0.01, size=1 + x.shape[1])
        self.errors_=[]
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(x,y):
                update = self.eta*(target-self.predict(xi))
                self.w_[1:]+= update*xi
                self.w_[0]+= update
                errors +=int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    def net_input(self, x):
        return np.dot(x, self.w_[1:]) + self.w_[0]
    
    def predict(self, x):
        return np.where(self.net_input(x) >= 0.0, 1, -1)

#Codigo Principal
df = pd.read_csv('iris.data', header=None, encoding='utf-8')
df.tail()
df.head()
print(df.head())
print(df.tail())
#Seleccion de setosa y versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
#Extraer largo sepal y largo petalo
x = df.iloc[0:100, [0, 2]].values
#Impresion de los datos
plt.scatter(x[:50, 0], x[:50, 1],color='red', marker='o', label='setosa')
plt.scatter(x[50:100, 0], x[50:100, 1],color='blue', marker='x', label='versicolor')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()
#Perceptron
ppn = perceptron(eta=0.1, n_iter=10)
ppn.fit(x , y)
plt.plot(range(1, len(ppn.errors_) + 1),ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()

from matplotlib.colors import ListedColormap

def plot_decision_regions(X,y,classifier,resolution=0.02):
    #configuracion marker generator y colormap
    markers=('s','x','o','^','v')
    colors=('red','blue','lightgreen','gray','cyan')
    cmap=ListedColormap(colors[:len(np.unique(y))])

    #plot superficie de desicion
    x1_min, x1_max=x[:,0].min()-1, x[:,0].max() +1
    x2_min, x2_max=x[:,1].min()-1, x[:,0].max() +1
    xx1, xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution), np.arange(x2_min,x2_max,resolution))
    z=classifier.predict(np.array( [xx1.ravel(), xx2.ravel() ]).T )
    z=z.reshape(xx1.shape)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    #plot clase ejemplo
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl, 0], 
                    y=X[y==cl, 1],
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')
plot_decision_regions(x, y, classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()
