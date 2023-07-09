# Tarea_Finla_TD_BAFM
 

Proyecto Final Tratamiento de Datos
Betty Florencia

Se pide:
1. Crear un repositorio de Github (publico) en el que se va a subir un jupyter notebook y un archivo
README.md (como mínimo)
2. Obtener un clasificador de imágenes de forma que dada una nueva imagen se pueda obtener la clase
correspondiente.
3. Se pide obtener las matrices de confusión del modelo, la matriz de confusión del error en training y la de
test.

# Desarrollo

1. Se realizó el repositorio de Github solicitado
2. Se desarrollo un clasificador de imagen

Como prerequisito necesitamos 

-Instalar libreria de pip install opencv-python

    !pip install opencv-python


Defaulting to user installation because normal site-packages is not writeable
Collecting opencv-python
  Downloading opencv_python-4.8.0.74-cp37-abi3-win_amd64.whl (38.1 MB)
     -------------------------------------- 38.1/38.1 MB 13.3 MB/s eta 0:00:00
Requirement already satisfied: numpy>=1.19.3 in c:\programdata\anaconda3\lib\site-packages (from opencv-python) (1.23.5)
Installing collected packages: opencv-python
Successfully installed opencv-python-4.8.0.74


Para el clasificador he usado el modelo secuencial con 5 capas de convoluciones

Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param    
=================================================================
 conv2d (Conv2D)             (None, 222, 222, 32)      896       
                                                                 
 max_pooling2d (MaxPooling2D  (None, 111, 111, 32)     0         
 )                                                               
                                                                 
 conv2d_1 (Conv2D)           (None, 109, 109, 64)      18496     
                                                                 
 flatten (Flatten)           (None, 760384)            0         
                                                                 
 output (Dense)              (None, 8)                 6083080   
                                                                 
=================================================================
Total params: 6,102,472
Trainable params: 6,102,472
Non-trainable params: 0
_________________________________________________________________

al compilar el modelo lo he optimizado con el "adadelta"

//El algoritmo Adadelta es un método de optimización basado en gradientes que se utiliza para ajustar los pesos y los sesgos del modelo durante el proceso de entrenamiento. Se caracteriza por adaptar automáticamente la tasa de aprendizaje en función del historial 
acumulado de los gradientes anteriores.

Código para entrenar el modelo - fit_generator() 

Para entrenar un modelo utilizando un generador de datos. 

model.fit_generator() es un método utilizado para entrenar un modelo utilizando generadores de datos en lugar de cargar todos los datos en memoria de una vez.

epochs es el número de épocas de entrenamiento, es decir, la cantidad de veces que el modelo verá el conjunto de datos completo durante el entrenamiento.

steps_per_epoch es el número de pasos que se realizarán en cada etapa del entrenamiento. Cada paso corresponde a una iteración a través de los datos.

validation_steps es similar a steps_per_epoch, pero para el conjunto de test. Especifica el número de pasos que se realizarán en cada etapa de test.



Utilicé este comando para guardar el modelo custom_model.save("model_secuencial.h5") y luego poder compararlo con el otro modelo Mobilenet 


3.
Comparé modelos Secuencial y Mobilenet

luego genero graficas de train y test 


Verifiqué con modelo Secuencial y con Mobilenet para encontrar coincidencia de imagenes y el resultado es distinto 

En Secuencial 0s 199ms/step CLASS_04; en Mobilenet 1s 558ms/step CLASS_05  

Conclusión: El modelo Secuencial tomo menos tiempo que Mobilenet


Luego corri el comando para generar las matrices pero me solicitó instalar una libreria 

---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
Cell In[21], line 4
      2 from sklearn.metrics import confusion_matrix, f1_score, roc_curve, precision_score, recall_score, accuracy_score, roc_auc_score
      3 from sklearn import metrics
----> 4 from mlxtend.plotting import plot_confusion_matrix
      5 from keras.models import load_model
      6 import numpy as np

ModuleNotFoundError: No module named 'mlxtend'

Intalé la libreria mlxtend

!pip install mlxtend
Defaulting to user installation because normal site-packages is not writeable
Collecting mlxtend
  Downloading mlxtend-0.22.0-py2.py3-none-any.whl (1.4 MB)
     ---------------------------------------- 1.4/1.4 MB 2.5 MB/s eta 0:00:00
Requirement already satisfied: matplotlib>=3.0.0 in c:\programdata\anaconda3\lib\site-packages (from mlxtend) (3.7.0)
Requirement already satisfied: setuptools in c:\programdata\anaconda3\lib\site-packages (from mlxtend) (65.6.3)
Requirement already satisfied: joblib>=0.13.2 in c:\programdata\anaconda3\lib\site-packages (from mlxtend) (1.1.1)
Requirement already satisfied: numpy>=1.16.2 in c:\programdata\anaconda3\lib\site-packages (from mlxtend) (1.23.5)
Requirement already satisfied: pandas>=0.24.2 in c:\programdata\anaconda3\lib\site-packages (from mlxtend) (1.5.3)
Requirement already satisfied: scikit-learn>=1.0.2 in c:\programdata\anaconda3\lib\site-packages (from mlxtend) (1.2.1)
Requirement already satisfied: scipy>=1.2.1 in c:\programdata\anaconda3\lib\site-packages (from mlxtend) (1.10.0)
Requirement already satisfied: python-dateutil>=2.7 in c:\programdata\anaconda3\lib\site-packages (from matplotlib>=3.0.0->mlxtend) (2.8.2)
Requirement already satisfied: fonttools>=4.22.0 in c:\programdata\anaconda3\lib\site-packages (from matplotlib>=3.0.0->mlxtend) (4.25.0)
Requirement already satisfied: kiwisolver>=1.0.1 in c:\programdata\anaconda3\lib\site-packages (from matplotlib>=3.0.0->mlxtend) (1.4.4)
Requirement already satisfied: pyparsing>=2.3.1 in c:\programdata\anaconda3\lib\site-packages (from matplotlib>=3.0.0->mlxtend) (3.0.9)
Requirement already satisfied: contourpy>=1.0.1 in c:\programdata\anaconda3\lib\site-packages (from matplotlib>=3.0.0->mlxtend) (1.0.5)
Requirement already satisfied: packaging>=20.0 in c:\programdata\anaconda3\lib\site-packages (from matplotlib>=3.0.0->mlxtend) (22.0)
Requirement already satisfied: pillow>=6.2.0 in c:\programdata\anaconda3\lib\site-packages (from matplotlib>=3.0.0->mlxtend) (9.4.0)
Requirement already satisfied: cycler>=0.10 in c:\programdata\anaconda3\lib\site-packages (from matplotlib>=3.0.0->mlxtend) (0.11.0)
Requirement already satisfied: pytz>=2020.1 in c:\programdata\anaconda3\lib\site-packages (from pandas>=0.24.2->mlxtend) (2022.7)
Requirement already satisfied: threadpoolctl>=2.0.0 in c:\programdata\anaconda3\lib\site-packages (from scikit-learn>=1.0.2->mlxtend) (2.2.0)
Requirement already satisfied: six>=1.5 in c:\programdata\anaconda3\lib\site-packages (from python-dateutil>=2.7->matplotlib>=3.0.0->mlxtend) (1.16.0)
Installing collected packages: mlxtend
Successfully installed mlxtend-0.22.0

Conclusión de matrices de confusion de ambos modelos

Los resultados muestran que el modelo Secuencial tiene un rendimiento muy bajo en todas las métricas, ya que la precisión, recall y f1-score son todos cercanos a cero. La precisión global del modelo es de 0.0790, lo que indica un rendimiento muy bajo.

Los resultados en el modelo Mobilenet muestran que las clases 2, 3, 4, 5 y 6 tienen valores de precisión mayores a cero, lo que indica que el modelo tiene más aciertos positivos en esas clases. La precisión global del modelo es del 58.52%