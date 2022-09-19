# Módulo 2 Uso de framework o biblioteca de aprendizaje máquina para la implementación de una solución. (Portafolio Implementación)
Dentro de esta carpeta se encuentran los siguientes archivos:
- `README.md`: Este archivo.
- `momento_2_data_analysis.ipynb`: Notebook conteniendo un análisis exploratorio de datos.
- `momento_2_model.py`: Script conteniendo la aplicacion de varios modelos de machine learning y algunas predicciones (Este es el archivo a revisar).
- `wine.data`: Dataset utilizado.
- `wine.names`: Descripción del dataset utilizado.

## Análisis exploratorio de datos
### Librerías utilizadas
- `pandas`: Para el manejo de datos.
- `matplotlib`: Para la visualización de datos.
- `sklearn`: Para el escalado de datos y la aplicacion de PCA.

### Descripción
El análisis exploratorio de datos se encuentra en el archivo `momento_2_data_analysis.ipynb`. En este se 
puede encontrar un pequeño análisis de los datos, buscando entenderlos y encontrar posibles relaciones entre
las variables. También se puede encontrar la aplicacion del escalamiento de los datos y reducción de dimensionalidad
con PCA para incrementar la precision de los modelos.

## Modelos de machine learning
### Librerías utilizadas
- `pandas`: Para el manejo de datos.
- `sklearn`: Para la aplicación de modelos de machine learning.
- `warnings`: Para ignorar los warnings de los modelos (Mayormente warnings de metodos deprecados).
- `lightgbm`: Para la aplicación de modelos de machine learning.
- `xgboost`: Para la aplicación de modelos de machine learning.

### Descripción
Los modelos de machine learning se encuentran en el archivo `momento_2_model.py`. En este se puede encontrar
la aplicación de varios modelos de machine learning, se utilizaron los siguientes modelos con el fin de encontrar
el mejor modelo para la predicción del tipo de vino:
- Arbol de decisión
- Bosque aleatorio
- K vecinos más cercanos
- Regresión logística
- SVM
- MLP
- XGBoost
- LightGBM

Con el fin de mejorar aun mas la precision de los modelos, se comparo la precision de los modelos con y sin PCA.
En este caso, aunque se puede observar que la precision de algunos modelos mejora con PCA, la precision de otros
modelos disminuye.

| Model               | Accuracy_normal | Accuracy_pca |
|---------------------|-----------------|--------------|
| Decision Tree       | 96.30 %         | 98.15 %      |
| Random Forest       | 100.00 %        | 98.15 %      |
| KNN                 | 96.30 %         | 98.15 %      |
| Logistic Regression | 98.15 %         | 98.15 %      |
| SVM                 | 98.15 %         | 98.15 %      |
| MLP                 | 98.15 %         | 98.15 %      |
| XGBoost             | 94.44 %         | 96.30 %      |
| LightGBM            | 98.15 %         | 96.30 %      |

Se utilizó un porcentaje de 70% para el entrenamiento y 30% para la prueba de los modelos.

### Predicciones
Para realizar las predicciones se utilizó el modelo de Random Forest, ya que este fue el modelo que obtuvo la mejor
precision. Se hicieron predicciones tanto con el modelo sin PCA como con el modelo con PCA para comparar las predicciones
de ambos modelos.

Se utilizaron los siguientes datos para realizar las predicciones los cuales son parte del dataset de prueba:

| Alcohol | Malic acid | Ash  | Alcalinity of ash | Magnesium | Total phenols | Flavanoids | Nonflavanoid phenols | Proanthocyanins | Color intensity | Hue  | OD280/OD315 of diluted wines | Proline |
|---------|------------|------|-------------------|-----------|---------------|------------|----------------------|-----------------|-----------------|------|------------------------------|---------|
| 13.64   | 3.1        | 2.56 | 15.2              | 116.0     | 2.7           | 3.03       | 0.17                 | 1.66            | 5.1             | 0.96 | 3.36                         | 845.0   |
| 14.21   | 4.04       | 2.44 | 18.9              | 111.0     | 2.85          | 2.65       | 0.3                  | 1.25            | 5.24            | 0.87 | 3.33                         | 1080.0  |
| 12.93   | 2.81       | 2.7  | 21.0              | 96.0      | 1.54          | 0.5        | 0.53                 | 0.75            | 4.6             | 0.77 | 2.31                         | 600.0   |
| 13.73   | 1.5        | 2.7  | 22.5              | 101.0     | 3.0           | 3.25       | 0.29                 | 2.38            | 5.7             | 1.19 | 2.71                         | 1285.0  |
| 12.37   | 1.17       | 1.92 | 19.6              | 78.0      | 2.11          | 2.0        | 0.27                 | 1.04            | 4.68            | 1.12 | 3.48                         | 510.0   |
| 14.3    | 1.92       | 2.72 | 20.0              | 120.0     | 2.8           | 3.14       | 0.33                 | 1.97            | 6.2             | 1.07 | 2.65                         | 1280.0  |
| 12.0    | 3.43       | 2.0  | 19.0              | 87.0      | 2.0           | 1.64       | 0.37                 | 1.87            | 1.28            | 0.93 | 3.05                         | 564.0   |
| 13.4    | 3.91       | 2.48 | 23.0              | 102.0     | 1.8           | 0.75       | 0.43                 | 1.41            | 7.3             | 0.7  | 1.56                         | 750.0   |
| 11.61   | 1.35       | 2.7  | 20.0              | 94.0      | 2.74          | 2.92       | 0.29                 | 2.49            | 2.65            | 0.96 | 3.26                         | 680.0   |
| 13.36   | 2.56       | 2.35 | 20.0              | 89.0      | 1.4           | 0.5        | 0.37                 | 0.64            | 5.6             | 0.7  | 2.47                         | 780.0   |

Se obtuvieron las siguientes predicciones utilizando Random Forest:

| Prediccion Normal | Prediccion con PCA | Prediccion Esperada |
|-------------------|--------------------|---------------------|
| 1                 | 1                  | 1                   |
| 1                 | 1                  | 1                   |
| 3                 | 3                  | 3                   |
| 1                 | 1                  | 1                   |
| 2                 | 2                  | 2                   |
| 1                 | 1                  | 1                   |
| 2                 | 2                  | 2                   |
| 3                 | 3                  | 3                   |
| 2                 | 2                  | 2                   |
| 3                 | 3                  | 3                   |





