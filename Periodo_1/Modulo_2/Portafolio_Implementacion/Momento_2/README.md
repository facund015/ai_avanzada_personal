# Momento de Retroalimentación: Módulo 2 Uso de framework o biblioteca de aprendizaje máquina para la implementación de una solución. (Portafolio Implementación)
Dentro de esta carpeta se encuentran los siguientes archivos:
- `README.md`: Este archivo.
- `momento_retro_2_data_analysis.ipynb`: Notebook conteniendo un análisis exploratorio de datos.
- `momento_retro_2_model.py`: Script conteniendo la aplicacion de varios modelos de machine learning y algunas predicciones (Este es el archivo a revisar).
- `wine.data`: Dataset utilizado.
- `wine.names`: Descripción del dataset utilizado.

## Análisis exploratorio de datos
### Librerías utilizadas
- `pandas`: Para el manejo de datos.
- `matplotlib`: Para la visualización de datos.
- `sklearn`: Para el escalado de datos y la aplicacion de PCA.

### Descripción
El análisis exploratorio de datos se encuentra en el archivo `momento_retro_2_data_analysis.ipynb`. En este se 
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
Los modelos de machine learning se encuentran en el archivo `momento_retro_2_model.py`. En este se puede encontrar
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

Se utilizaron los siguientes datos para realizar las predicciones:

| Caso | Alcohol |  Malic   acid |  Ash |  Alcalinity of ash |  Magnesium |  Total   phenols |  Flavanoids |  Nonflavanoid phenols |  Proanthocyanins |  Color   intensity |  Hue |  OD280/OD315 of diluted wines |  Proline |
|------|---------|---------------|------|--------------------|------------|------------------|-------------|-----------------------|------------------|--------------------|------|-------------------------------|----------|
| 1    | 13.2    | 2.77          | 2.51 | 18.5               | 96         | 1.04             | 0.6         | 0.06                  | 0.96             | 5.28               | 0.93 | 3.05                          | 564      |
| 2    | 12.37   | 1.07          | 2.1  | 18.5               | 88         | 3.52             | 3.75        | 0.24                  | 1.95             | 4.5                | 1.04 | 2.77                          | 660      |
| 3    | 13.88   | 1.89          | 2.59 | 15                 | 101        | 3.25             | 3.56        | 0.17                  | 1.7              | 5.43               | 0.88 | 3.56                          | 1095     |
| 4    | 12.08   | 1.39          | 2.5  | 22.5               | 84         | 2.56             | 2.29        | 0.43                  | 1.04             | 2.9                | 0.93 | 3.19                          | 385      |
| 5    | 12.08   | 1.83          | 2.32 | 18.5               | 81         | 1.6              | 0.6         | 0.53                  | 1.55             | 3.2                | 1.08 | 2.27                          | 480      |
| 6    | 13.08   | 1.13          | 2.51 | 24                 | 78         | 2                | 1.58        | 0.4                   | 1.4              | 2.2                | 1.31 | 2.72                          | 630      |
| 7    | 12.67   | 1.39          | 2.5  | 22.5               | 84         | 2.56             | 2.29        | 0.43                  | 1.04             | 2.9                | 0.93 | 3.19                          | 385      |
| 8    | 14.08   | 1.83          | 2.32 | 18.5               | 81         | 1.6              | 0.6         | 0.53                  | 1.55             | 3.2                | 1.08 | 2.27                          | 480      |
| 9    | 11.38   | 1.13          | 2.51 | 24                 | 78         | 2                | 1.58        | 0.4                   | 1.4              | 2.2                | 1.31 | 2.72                          | 630      |
| 10   | 12.08   | 1.39          | 2.5  | 22.5               | 84         | 2.56             | 2.29        | 0.43                  | 1.04             | 2.9                | 0.93 | 3.19                          | 385      |

Se obtuvieron las siguientes predicciones utilizando Random Forest:

| Caso | Prediccion_normal | Prediccion_pca |
|------|-------------------|----------------|
| 1    | 3                 | 2              |
| 2    | 2                 | 2              |
| 3    | 1                 | 1              |
| 4    | 2                 | 2              |
| 5    | 2                 | 2              |
| 6    | 2                 | 2              |
| 7    | 2                 | 2              |
| 8    | 2                 | 2              |
| 9    | 2                 | 2              |
| 10   | 2                 | 2              |

Cabe mencionar que para realizar las predicciones, se utilizaran datos generados aleatoriamente en base a los datos 
originales, dado que no se cuentan con datos nuevos para realizar las predicciones. Esto significa que los datos
generados aleatoriamente no son necesariamente representativos de la realidad, pero se utilizan para mostrar el 
funcionamiento del modelo de Random Forest





