# Tarea 1  **[DeadLine: 1 dic 2020]**
*Nicolás Carrillo Sepulveda - Ingenieria Civil Telematica*
<br><br>
## Regresion Logistica Multiple
<br>

### La regresion losgistica multiple es una generalización de la regresion logistica ya estudiada. En esta su variable dependiente no esta restringida a solo dos categorias. 

### Por ejemplo, un estudio de cine necesita saber cual de sus peliculas sera la mas vista. Para esto usa una regresion logistica multiple o multinomial para determinar si la edad, el genero y las relaciones de pareja de cada persona sobre la pelicula que les gusta.
### Con este ejemplo se puede observar que la "salida" en el proceso de regresion logistica multiple tendra mas de dos, y estara "alimentado" por distinto tipos de parametros


## 2. Identificar otra base de datos mas grande que iris y correr LR para CLASIFICAR utilizando:
<br>
 *Explicar la base de datos*  
<BR>

<br><br><br>

### **Tabla NewtonCg-Multinomial**
| **C** | **solver** | **multiclass** | **Exactitud %** | 
| :------- | :------- | :------- | :------- |
|   1 | newton-cg | multinomial   |      91.2281 |
|   5 | newton-cg | multinomial   |      91.2281 |
|  10 | newton-cg | multinomial   |      91.2281 |
|  20 | newton-cg | multinomial   |      91.2281 |
|  50 | newton-cg | multinomial   |      91.2281 |
| 100 | newton-cg | multinomial   |      91.2281 |
| 500 | newton-cg | multinomial   |      91.2281 |
<br>

![](gif/newton_Multinomial.gif)
#### *figura: Animación de regresion logistica con solver "Newton-cg" y multiclass "multinomial" donde C varia de 1 a 500.*

*AGREGAR COMENTARIOS* *
<br><br>

### **Tabla NewtonCg-Ovr** 
| **C** | **solver** | **multiclass** | **Exactitud %** |
| :------- | :------- | :------- | :------- |
|   1 | newton-cg | ovr           |      91.2281 |
|   5 | newton-cg | ovr           |      91.2281 |
|  10 | newton-cg | ovr           |      91.2281 |
|  20 | newton-cg | ovr           |      91.2281 |
|  50 | newton-cg | ovr           |      91.2281 |
| 100 | newton-cg | ovr           |      91.2281 |
| 500 | newton-cg | ovr           |      91.2281 |
<br>

![](gif/newton_ovr.gif)

#### *figura: Animación de regresion logistica con solver "Newton-cg" y multiclass "Ovr" donde C varia de 1 a 500.*


*AGREGAR COMENTARIOS* *
<br><br>

### **Tabla liblinear-Ovr** 
| **C** | **solver** | **multiclass** | **Exactitud %** |
| :------- | :------- | :------- | :------- |
|   1 | liblinear | ovr           |      91.2281 |
|   5 | liblinear | ovr           |      91.2281 |
|  10 | liblinear | ovr           |      91.2281 |
|  20 | liblinear | ovr           |      91.2281 |
|  50 | liblinear | ovr           |      91.2281 |
| 100 | liblinear | ovr           |      91.2281 |
| 500 | liblinear | ovr           |      91.2281 |
<br>

![](gif/liblinear_ovr.gif)
#### *figura: Animación de regresion logistica con solver "liblinear" y multiclass "ovr" donde C varia de 1 a 500.*

*AGREGAR COMENTARIOS* *
<br><br>


### **Tabla Sag-Multinomial**
| **C** | **solver** | **multiclass** | **Exactitud %** |
| :------- | :------- | :------- | :------- |
|   1 | sag       | multinomial   |      91.2281 |
|   5 | sag       | multinomial   |      91.2281 |
|  10 | sag       | multinomial   |      91.2281 |
|  20 | sag       | multinomial   |      91.2281 |
|  50 | sag       | multinomial   |      91.2281 |
| 100 | sag       | multinomial   |      91.2281 |
| 500 | sag       | multinomial   |      91.2281 |
<br>

![](gif/sag_Multinomial.gif)
#### *figura: Animación de regresion logistica con solver "Sag" y multiclass "Multinomial" donde C varia de 1 a 500.*


*AGREGAR COMENTARIOS* *
<br><br>


### **Tabla Sag-Ovr**
| **C** | **solver** | **multiclass** | **Exactitud %** |
| :------- | :------- | :------- | :------- |
|   1 | sag       | ovr           |      91.2281 |
|   5 | sag       | ovr           |      91.2281 |
|  10 | sag       | ovr           |      91.2281 |
|  20 | sag       | ovr           |      91.2281 |
|  50 | sag       | ovr           |      91.2281 |
| 100 | sag       | ovr           |      91.2281 |
| 500 | sag       | ovr           |      91.2281 |
<br>

![](gif/sag_ovr.gif)
#### *figura: Animación de regresion logistica con solver "sag" y multiclass "ovr" donde C varia de 1 a 500.*

*AGREGAR COMENTARIOS* *
<br><br>


### **Tabla Saga-Multinomial**
| **C** | **solver** | **multiclass** | **Exactitud %** |
| :------- | :------- | :------- | :------- |
|   1 | saga      | multinomial   |      91.2281 |
|   5 | saga      | multinomial   |      91.2281 |
|  10 | saga      | multinomial   |      91.2281 |
|  20 | saga      | multinomial   |      91.2281 |
|  50 | saga      | multinomial   |      91.2281 |
| 100 | saga      | multinomial   |      91.2281 |
| 500 | saga      | multinomial   |      91.2281 |
<br>

![](gif/saga_Multinomial.gif)
#### *figura: Animación de regresion logistica con solver "Saga" y multiclass "Multinomial" donde C varia de 1 a 500.*

*AGREGAR COMENTARIOS* *
<br><br>


### **Tabla Saga-Ovr**
| **C** | **solver** | **multiclass** | **Exactitud %** |
| :------- | :------- | :------- | :------- |
|   1 | saga      | ovr           |      91.2281 |
|   5 | saga      | ovr           |      91.2281 |
|  10 | saga      | ovr           |      91.2281 |
|  20 | saga      | ovr           |      91.2281 |
|  50 | saga      | ovr           |      91.2281 |
| 100 | saga      | ovr           |      91.2281 |
| 500 | saga      | ovr           |      91.2281 |
<br>

![](gif/saga_ovr.gif)
#### *figura: Animación de regresion logistica con solver "Sag" y multiclass "Multinomial" donde C varia de 1 a 500.*


 *AGREGAR COMENTARIOS* *
<br><br>

En ninguno de los casos analizados se ve presencia de ovefitting en ninguno de los casos. Pero se podria plantear que al ser un metodo lineal existiria undefitting. (aunque los porcentajes son bastante altos en algunos casos)