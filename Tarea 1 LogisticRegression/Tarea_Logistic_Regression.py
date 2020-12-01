# Importar la info
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

cancer = datasets.load_breast_cancer()
X = cancer.data[:,[0,1]]
y = cancer.target
print('-------------------------------------------------')
print('Tipos de tumor: ',list(cancer.target_names))
print('Caracteristicas de tumor: ')
print(list(cancer.feature_names))
print('-------------------------------------------------')
#Separar datos de entrenamiento y prueba.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.1, random_state=1, stratify=y)

#Escalar los datos
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()               # Estandarizar características eliminando la media y escalando a la varianza de la unidad
sc.fit(X_train)                     # Calcule la media y la estándar que se utilizarán para escalar posteriormente.
X_train_std = sc.transform(X_train) # Realice la estandarización centrando y escalando
X_test_std = sc.transform(X_test)

# Definicion del algoritmo a utilizar "Logistic Regression"
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from tabulate import tabulate
import warnings

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    plt.figure(figsize=(19,10),dpi=100) 
    #tamaño grafico
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



X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
warnings.filterwarnings("ignore")

""" solver {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'}, predeterminado = 'lbfgs'  Algoritmo a utilizar en el problema de optimización.
Para conjuntos de datos pequeños, 'liblinear' es una buena opción, mientras que 'sag' y 'saga' son más rápidos para los grandes.
Para problemas multiclase, solo 'newton-cg', 'sag', 'saga' y 'lbfgs' manejan la pérdida multinomial; 'liblinear' se limita a esquemas uno versus resto.
'newton-cg', 'lbfgs', 'sag' y 'saga' manejan L2 o sin penalización
'liblinear' y 'saga' también manejan la penalización L1
'saga' también admite la penalización de 'elasticnet'
'liblinear' no admite la configuración penalty='none' """

""" multi_class {'auto', 'ovr', 'multinomial'}, predeterminado = 'auto'
Si la opción elegida es 'ovr', entonces se ajusta un problema binario para cada etiqueta. 
Para 'multinomial', la pérdida minimizada es el ajuste de pérdida multinomial en toda la distribución de probabilidad, incluso cuando los datos son binarios. 
'multinomial' no está disponible cuando solver = 'liblinear'.
'auto' selecciona 'ovr' si los datos son binarios, o si solver = 'liblinear', y de lo contrario selecciona 'multinomial'.
 """

solv   = ['newton-cg', 'liblinear', 'sag', 'saga']
mclass = ['multinomial','ovr']# 'auto',
table = []
C =[1,10,50,100,500,750,1000]
for s in solv:
    for mc in mclass:
        for c in C:
            if mc=='multinomial':
                if s=='liblinear':
                    break
            lr = LogisticRegression(C=c, random_state =42,solver=s,multi_class=mc)
            lr.fit(X_train_std, y_train)
            y_pred = lr.predict(X_test_std)
            plot_decision_regions(X_combined_std,y_combined,classifier=lr,test_idx=range(0, 57))
            plt.legend(loc='upper left')
            plt.tight_layout()
            plt.suptitle('C='+str(c)+' Solver='+str(s)+' Multicass='+str(mc),fontsize=20)
            #plt.show()
            plt.savefig('img/lr_'+str(s)+'_'+str(mc)+'_'+str(c)+'.png')
            print('creando imagen: img/lr_'+str(s)+'_'+str(mc)+'_'+str(c)+'.png')
            #Verificacion
            table.append([c,s,mc,"{:.4f}".format(accuracy_score(y_test,y_pred)*100)])
print('')
print('')
print(tabulate(table, headers = ['C','solver', ' multiclass', ' Exactitud'], tablefmt="github"))
print('-------------------------------------------------')               
 