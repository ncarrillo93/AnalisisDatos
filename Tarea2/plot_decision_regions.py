import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
#Clase perceptron
ws = []
class perceptron(object):
    def __init__(self,eta=0.01,n_iter=50,random_state=1):
        self.eta=eta
        self.n_iter= n_iter
        self.random_state=random_state
    
    def fit(self, x, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_=rgen.normal(loc=0.0, scale= 0.01, size=1 + x.shape[1])
        ws.append(rgen.normal(loc=0.0, scale= 0.01, size=1 + x.shape[1]))
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

def plot_decision_regions(X, y, clasiffier, resolution=0.02) :
    #Setup market generator and colormap
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue','lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y)) ] )

    # plot the decision surface
    x1_min, x1_max = X[:,0].min() -1, X[:, 0].max() + 1
    x2_min, x2_max = X[:,1].min() -1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = clasiffier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())

    #plot class example
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y== cl,0], y = X[y == cl, 1], alpha=0.8, c=colors[idx], marker=markers[idx], label=cl, edgecolor='black')

#Codigo Principal - Largo de sepalo, ancho de sepalo, largo de petalo, ancho de petalo, especie
df = pd.read_csv('Tarea1/iris.data', header=None, encoding='utf-8')
y   = df.iloc[0:100, 4].values
y   = np.where(y == 'Iris-setosa', -1, 1)


#Extraer largo de sepalo y ancho de sepalo
x   = df.iloc[0:100, [0,2]].values

ppn = perceptron(eta=0.1, n_iter=7)
ppn.fit(x , y)
plt.plot(range(1, len(ppn.errors_) +1), ppn.errors_,marker='o')
plt.xlabel('Epocas')
plt.ylabel('numero de actualizaciones')
plt.show()

plot_decision_regions(x, y, clasiffier=ppn)
plt.xlabel('Largo Sepalo [cm]')
plt.xlabel('Largo Sepalo [cm]')
plt.legend(loc='upper left')
plt.show()
         
print(str(ws))
