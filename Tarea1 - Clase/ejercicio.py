import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
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

#Codigo Principal - Largo de sepalo, ancho de sepalo, largo de petalo, ancho de petalo, especie
df = pd.read_csv('Tarea1/iris.data', header=None, encoding='utf-8')

y   = df.iloc[0:150, 4].values
y   = np.where(y == 'Iris-setosa', -1, 1)

y_1 = df.iloc[0:150, 4].values 
y_1 = np.where(y_1 =='Iris-versicolor',-1,1)

y_2 = df.iloc[0:150, 4].values
y_2 = np.where(y_2 =='Iris-virginica', -1, 1) 

#Extraer largo de sepalo y ancho de sepalo
x   = df.iloc[0:150, [0,1]].values
x_1 = df.iloc[0:150, [0,2]].values
x_2 = df.iloc[0:150, [0,3]].values
x_3 = df.iloc[0:150, [1,2]].values
x_4 = df.iloc[0:150, [1,3]].values
x_5 = df.iloc[0:150, [2,3]].values
#- Largo de sepalo, ancho de sepalo, largo de petalo, ancho de petalo, especie
print(x[0])
print(x_1[0])
print(x_2[0])
print(x_3[0])
print(x_4[0])
print(x_5[0])

#Impresion de los datos largo de sepalo y ancho de sepalo
plt.figure(figsize=(14,10)) #tamaño grafico
plt.subplot(231)
plt.scatter(x[:50, 0]   , x[:50    , 1], color='red'  , marker='o', label='setosa'    )
plt.scatter(x[50:100, 0], x[50:100 , 1], color='blue' , marker='o', label='versicolor')
plt.scatter(x[100:150,0], x[100:150, 1], color='green', marker='o', label='virginica' )
plt.xlabel('largo sepalo [cm]')
plt.ylabel('ancho sepalo [cm]')
plt.legend(loc='upper left')

plt.subplot(232)
plt.scatter(x_1[:50, 0]   , x_1[:50    , 1], color='red'  , marker='o', label='setosa'    )
plt.scatter(x_1[50:100, 0], x_1[50:100 , 1], color='blue' , marker='o', label='versicolor')
plt.scatter(x_1[100:150,0], x_1[100:150, 1], color='green', marker='o', label='virginica' )
plt.xlabel('largo sepalo [cm]')
plt.ylabel('largo petalo [cm]')
plt.legend(loc='upper left')

plt.subplot(233)
plt.scatter(x_2[:50, 0]   , x_2[:50    , 1], color='red'  , marker='o', label='setosa'    )
plt.scatter(x_2[50:100, 0], x_2[50:100 , 1], color='blue' , marker='o', label='versicolor')
plt.scatter(x_2[100:150,0], x_2[100:150, 1], color='green', marker='o', label='virginica' )
plt.xlabel('largo sepalo [cm]')
plt.ylabel('ancho petalo [cm]')
plt.legend(loc='upper left')

plt.subplot(234)
plt.scatter(x_3[:50, 0]   , x_3[:50    , 1], color='red'  , marker='o', label='setosa'    )
plt.scatter(x_3[50:100, 0], x_3[50:100 , 1], color='blue' , marker='o', label='versicolor')
plt.scatter(x_3[100:150,0], x_3[100:150, 1], color='green', marker='o', label='virginica' )
plt.xlabel('ancho de sepalo [cm]')
plt.ylabel('largo de petalo [cm]') 
plt.legend(loc='upper left')

plt.subplot(235)
plt.scatter(x_4[:50, 0]   , x_4[:50    , 1], color='red'  , marker='o', label='setosa'    )
plt.scatter(x_4[50:100, 0], x_4[50:100 , 1], color='blue' , marker='o', label='versicolor')
plt.scatter(x_4[100:150,0], x_4[100:150, 1], color='green', marker='o', label='virginica' )
plt.xlabel('ancho sepalo [cm]')
plt.ylabel('ancho petalo [cm]')
plt.legend(loc='upper left')

plt.subplot(236)
plt.scatter(x_5[:50, 0]   , x_5[:50    , 1], color='red'  , marker='o', label='setosa'    )
plt.scatter(x_5[50:100, 0], x_5[50:100 , 1], color='blue' , marker='o', label='versicolor')
plt.scatter(x_5[100:150,0], x_5[100:150, 1], color='green', marker='o', label='virginica' )
plt.xlabel('largo petalo [cm]')
plt.ylabel('ancho petalo [cm]')
plt.legend(loc='upper left')
plt.savefig('Tarea1/img/trios.png')
plt.show()

#Perceptron 
plt.figure(figsize=(14,10))
plt.subplot(231)
plt.title("Perceptron Setosa - Versicolor")
ppn = perceptron(eta=0.1, n_iter=10)
ppn.fit(x , y)
plt.plot(range(1, len(ppn.errors_) + 1),ppn.errors_, marker='o')
plt.xlabel('Epocas')
plt.ylabel('Numero of actualizaciones')

plt.subplot(232)
ppn = perceptron(eta=0.1, n_iter=10)
ppn.fit(x_1 , y)
plt.plot(range(1, len(ppn.errors_) + 1),ppn.errors_, marker='o')
plt.xlabel('Epocas')
plt.ylabel('Numero of actualizaciones')

plt.subplot(233)
ppn = perceptron(eta=0.1, n_iter=10)
ppn.fit(x_2 , y)
plt.plot(range(1, len(ppn.errors_) + 1),ppn.errors_, marker='o')
plt.xlabel('Epocas')
plt.ylabel('Numero of actualizaciones')

plt.subplot(234)
ppn = perceptron(eta=0.1, n_iter=10)
ppn.fit(x_3 , y)
plt.plot(range(1, len(ppn.errors_) + 1),ppn.errors_, marker='o')
plt.xlabel('Epocas')
plt.ylabel('Numero of actualizaciones')

plt.subplot(235)
ppn = perceptron(eta=0.1, n_iter=10)
ppn.fit(x_4 , y)
plt.plot(range(1, len(ppn.errors_) + 1),ppn.errors_, marker='o')
plt.xlabel('Epocas')
plt.ylabel('Numero of actualizaciones')

plt.subplot(236)
ppn = perceptron(eta=0.1, n_iter=10)
ppn.fit(x_5 , y)
plt.plot(range(1, len(ppn.errors_) + 1),ppn.errors_, marker='o')
plt.xlabel('Epocas')
plt.ylabel('Numero of actualizaciones')
plt.savefig('Tarea1/img/perceptron_setosa_versicolor.png')
plt.show()

#Perceptron
# tamaño grafico
plt.figure(figsize=(14,10))
plt.subplot(231)
plt.title("Perceptron Setosa - Virginica")
ppn = perceptron(eta=0.1, n_iter=10)
ppn.fit(x , y_1)
plt.plot(range(1, len(ppn.errors_) + 1),ppn.errors_, marker='o')
plt.xlabel('Epocas')
plt.ylabel('Numero of actualizaciones')

plt.subplot(232)
ppn = perceptron(eta=0.1, n_iter=10)
ppn.fit(x_1 , y_1)
plt.plot(range(1, len(ppn.errors_) + 1),ppn.errors_, marker='o')
plt.xlabel('Epocas')
plt.ylabel('Numero of actualizaciones')

plt.subplot(233)
ppn = perceptron(eta=0.1, n_iter=10)
ppn.fit(x_2 , y_1)
plt.plot(range(1, len(ppn.errors_) + 1),ppn.errors_, marker='o')
plt.xlabel('Epocas')
plt.ylabel('Numero of actualizaciones')

plt.subplot(234)
ppn = perceptron(eta=0.1, n_iter=10)
ppn.fit(x_3 , y_1)
plt.plot(range(1, len(ppn.errors_) + 1),ppn.errors_, marker='o')
plt.xlabel('Epocas')
plt.ylabel('Numero of actualizaciones')

plt.subplot(235)
ppn = perceptron(eta=0.1, n_iter=10)
ppn.fit(x_4 , y_1)
plt.plot(range(1, len(ppn.errors_) + 1),ppn.errors_, marker='o')
plt.xlabel('Epocas')
plt.ylabel('Numero of actualizaciones')

plt.subplot(236)
ppn = perceptron(eta=0.1, n_iter=10)
ppn.fit(x_5 , y_1)
plt.plot(range(1, len(ppn.errors_) + 1),ppn.errors_, marker='o')
plt.xlabel('Epocas')
plt.ylabel('Numero of actualizaciones')
plt.savefig('Tarea1/img/perceptron_setosa_virginica.png')
plt.show()

#Percerptron versicolor-virginica
plt.figure(figsize=(14,10))
plt.subplot(231)
plt.title("Perceptron versicolor-virginica")
ppn = perceptron(eta=0.1, n_iter=10)
ppn.fit(x , y_2)
plt.plot(range(1, len(ppn.errors_) + 1),ppn.errors_, marker='o')
plt.xlabel('Epocas')
plt.ylabel('Numero of actualizaciones')

plt.subplot(232)
ppn = perceptron(eta=0.1, n_iter=10)
ppn.fit(x_1 , y_2)
plt.plot(range(1, len(ppn.errors_) + 1),ppn.errors_, marker='o')
plt.xlabel('Epocas')
plt.ylabel('Numero of actualizaciones')

plt.subplot(233)
ppn = perceptron(eta=0.1, n_iter=10)
ppn.fit(x_2 , y_2)
plt.plot(range(1, len(ppn.errors_) + 1),ppn.errors_, marker='o')
plt.xlabel('Epocas')
plt.ylabel('Numero of actualizaciones')

plt.subplot(234)
ppn = perceptron(eta=0.1, n_iter=10)
ppn.fit(x_3 , y_2)
plt.plot(range(1, len(ppn.errors_) + 1),ppn.errors_, marker='o')
plt.xlabel('Epocas')
plt.ylabel('Numero of actualizaciones')

plt.subplot(235)
ppn = perceptron(eta=0.1, n_iter=10)
ppn.fit(x_4 , y_2)
plt.plot(range(1, len(ppn.errors_) + 1),ppn.errors_, marker='o')
plt.xlabel('Epocas')
plt.ylabel('Numero of actualizaciones')

plt.subplot(236)
ppn = perceptron(eta=0.1, n_iter=10)
ppn.fit(x_5 , y_2)
plt.plot(range(1, len(ppn.errors_) + 1),ppn.errors_, marker='o')
plt.xlabel('Epocas')
plt.ylabel('Numero of actualizaciones')
plt.savefig('Tarea1/img/perceptron_versicolor_virginica.png')
plt.show()


