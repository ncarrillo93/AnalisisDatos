# Importar la info
from sklearn import datasets
cancer = datasets.load_breast_cancer()
print('Tipos de tumor: ',list(cancer.target_names))
X = cancer.data
y = cancer.target

#Separar datos de entrenamiento y prueba.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=1)

#Escalar los datos
from sklearn.preprocessing import StandardScaler
escalar = StandardScaler()               # Estandarizar características eliminando la media y escalando a la varianza de la unidad
escalar.fit(X_train)                     # Calcule la media y la estándar que se utilizarán para escalar posteriormente.
X_train_std = escalar.transform(X_train) # Realice la estandarización centrando y escalando
X_test_std  = escalar.transform(X_test)

# Definicion del algoritmo a utilizar "Logistic Regression"
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from tabulate import tabulate
import warnings

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
mclass = ['auto', 'ovr' , 'multinomial']# , 'multinomial'
table = []
warnings.filterwarnings("ignore")

for s in solv:
    for mc in mclass:
        if mc=='multinomial':
            if s=='liblinear':
                break
        lr = LogisticRegression(C=100, random_state =1,solver=s,multi_class=mc)
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        #Verificacion
        table.append([s,mc,"{:.3f}".format(precision_score(y_test, y_pred)*100),"{:.3f}".format(accuracy_score(y_test,y_pred)*100)])
print(tabulate(table, headers = ['solver', 'multiclass', 'Precision', 'Exactitud']))       
pd.write_csv()