import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn import datasets 

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02) :

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
    if test_idx:
        #plot all examples
        X_test, y_test, = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:,1], c='', edgecolor='black', alpha=1.0, linewidth=1, marker='o',s=100 ,label='test set')

# Importar la info
cancer = datasets.load_breast_cancer()
# Definicion de los datos correspondientes a las etiquetas
X = cancer.data
y = cancer.target
## Implementacion de regresion logistica
    # Separar los datos de "Train" en entrenamiento y prueba para probar los algoritmos  
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    #Escalar todos los datos
from sklearn.preprocessing import StandardScaler
escalar = StandardScaler()
escalar.fit(X_train)
X_train_std = escalar.transform(X_train)
X_test_std = escalar.transform(X_test)

## Definicion del algoritmo a utilizar LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
LogisticRegression = LogisticRegression(C=100, random_state=1, solver='newton-cg',multi_class='auto')
#entrenamiento del modelo
LogisticRegression.fit(X_train,y_train)
y_pred = LogisticRegression.predict(X_test)
#Verificacion
from sklearn.metrics import confusion_matrix
matriz = confusion_matrix(y_test, y_pred)
print('Matriz de confusion:',matriz)

#Calculo de precision del modelo
from sklearn.metrics import precision_score
precision = precision_score(y_test, y_pred)
print('Precision del Modelo',precision)

#Calculo exactitud modelo
from sklearn.metrics import accuracy_score
exactitud = accuracy_score(y_test,y_pred)
print('Exactitud del Modelo', exactitud)



