# Tarea 1  **[DeadLine: 1 dic 2020]**
*Nicolás Carrillo Sepulveda - Ingenieria Civil Telematica*
<br><br>
## Regresion Logistica Multiple
<br>

### La regresion losgistica multiple es una generalización de la regresion logistica ya estudiada. En esta su variable dependiente no esta restringida a solo dos categorias. 

### Por ejemplo, un estudio de cine necesita saber cual de sus peliculas sera la mas vista. Para esto usa una regresion logistica multiple o multinomial para determinar si la edad, el genero y las relaciones de pareja de cada persona sobre la pelicula que les gusta.
### Con este ejemplo se puede observar que la "salida" en el proceso de regresion logistica multiple tendra mas de dos, y estara "alimentado" por distinto tipos de parametros ###

<br><br>

## 2. Identificar otra base de datos mas grande que iris y correr LR para CLASIFICAR utilizando:

<br>

 ### La base de datos utilizada es acerca de cancer de mama. Este posee datos de tumores, que tienen diez caracteristicas para cada tumor. En este caso se utilizo el radio promedio y el peor radio.  ###
 
<br><br>

### **Graficos Lbfgs-Multinomial**

[1]: gif/lbfgs-multinomial.gif
[2]: img2/lr_lbfgs_multinomial.png
![][1] ![][2] =50x10

#### *figura: Animación de regresion logistica con solver "Lbfgs" y multiclass "Multinomial".*

<br>

### **Graficos Lbfgs-Ovr**
<br>

[3]: gif/lbfgs-ovr.gif
[4]: img2/lr_lbfgs_ovr.png
![][3] ![][4] =50x10

#### *figura: Animación de regresion logistica con solver "Lbfgs" y multiclass "ovr".*

<br>

### **Graficos Liblinear-Ovr**
<br>

[5]: gif/liblinear-ovr.gif
[6]: img2/lr_liblinear_ovr.png
![][5] ![][6] =50x10

#### *figura: Animación de regresion logistica con solver "Liblinear" y multiclass "ovr".*

<br>

### **Tabla NewtonCg-Multinomial**
<br>

[7]: gif/newton-multinomial.gif
[8]: img2/lr_newton-cg_multinomial.png
![][7] ![][8] =50x10

#### *figura: Animación de regresion logistica con solver "Newton-cg" y multiclass "multinomial".*

<br>

### **Tabla NewtonCg-Ovr** 
<br>

[9]: gif/newton-ovr.gif
[10]: img2/lr_newton-cg_ovr.png
![][9] ![][10] =50x10

#### *figura: Animación de regresion logistica con solver "Newton-cg" y multiclass "Ovr".*

<br>

### **Grafico Sag-Ovr**
<br>

[9]: gif/sag-ovr.gif
[10]: img2/lr_sag_ovr.png
![][9] ![][10] =50x10

#### *figura: Animación de regresion logistica con solver "sag" y multiclass "ovr".*


### **Grafico Sag-Multinomial**
<br>

[11]: gif/sag-multinomial.gif
[12]: img2/lr_sag_multinomial.png
![][11] ![][12] =50x10

#### *figura: Animación de regresion logistica con solver "Sag" y multiclass "Multinomial".*

<br>

### **Grafico Saga-Multinomial**
<br>

[11]: gif/saga-multinomial.gif
[12]: img2/lr_saga_multinomial.png
![][11] ![][12] =50x10

#### *figura: Animación de regresion logistica con solver "Saga" y multiclass "Multinomial".*

### **Grafico Saga-Ovrl**
<br>

[11]: gif/saga-Ovr.gif
[12]: img2/lr_saga_Ovr.png
![][11] ![][12] =50x10

#### *figura: Animación de regresion logistica con solver "Saga" y multiclass "Ovr".*
<br>

 ###  **<u>Comentarios</u> :** 

 - Para poder observar la la precisión en el grafico animado. se genero un arreglo con diversos vaolres de C entre 0 a 1, y al final valores entre 1 y 5000.

 - Se puede observar que la precisión nunca llega a un 100% en ningun caso. Porque el metodo de regresión logica no llega nunca a separar todos los "puntos" en la grafica.

 - Si bien ninguna combinación llega a converger. Los resultados muestran que separa relativamente bien los dos grupos de tumores (Malgino y Benigno)

<br>

## Overfitting?
- En ninguno de los casos analizados se ve presencia de ovefitting en ninguno de los casos. Pero se podria plantear que al ser un metodo lineal existiria undefitting. (aunque los porcentajes son bastante altos en algunos casos)