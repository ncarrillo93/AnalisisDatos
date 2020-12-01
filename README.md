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
<br>

![](gif/newton_Multinomial.gif)
#### *figura: Animación de regresion logistica con solver "Newton-cg" y multiclass "multinomial" donde C varia de 1 a 500.*

*AGREGAR COMENTARIOS* *
<br><br>

### **Tabla NewtonCg-Ovr** 
<br>

![](gif/newton_ovr.gif)

#### *figura: Animación de regresion logistica con solver "Newton-cg" y multiclass "Ovr" donde C varia de 1 a 500.*


*AGREGAR COMENTARIOS* *
<br><br>

### **Tabla liblinear-Ovr** 
<br>

![](gif/liblinear_ovr.gif)
#### *figura: Animación de regresion logistica con solver "liblinear" y multiclass "ovr" donde C varia de 1 a 500.*

*AGREGAR COMENTARIOS* *
<br><br>


### **Tabla Sag-Multinomial**
<br>

![](gif/sag_Multinomial.gif)
#### *figura: Animación de regresion logistica con solver "Sag" y multiclass "Multinomial" donde C varia de 1 a 500.*


*AGREGAR COMENTARIOS* *
<br><br>


### **Tabla Sag-Ovr**
<br>

![](gif/sag_ovr.gif)
#### *figura: Animación de regresion logistica con solver "sag" y multiclass "ovr" donde C varia de 1 a 500.*

*AGREGAR COMENTARIOS* *
<br><br>


### **Tabla Saga-Multinomial**
<br>

![](gif/saga_Multinomial.gif)
#### *figura: Animación de regresion logistica con solver "Saga" y multiclass "Multinomial" donde C varia de 1 a 500.*

*AGREGAR COMENTARIOS* *
<br><br>

### **Tabla Saga-Ovr**
<br>

![](gif/saga_ovr.gif)
#### *figura: Animación de regresion logistica con solver "Sag" y multiclass "Multinomial" donde C varia de 1 a 500.*


 *AGREGAR COMENTARIOS* *
<br><br>

En ninguno de los casos analizados se ve presencia de ovefitting en ninguno de los casos. Pero se podria plantear que al ser un metodo lineal existiria undefitting. (aunque los porcentajes son bastante altos en algunos casos)
