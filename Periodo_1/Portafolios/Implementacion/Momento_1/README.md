# Módulo 2 Implementación de una técnica de aprendizaje máquina sin el uso de un framework. (Portafolio Implementación)
Dentro de esta carpeta se encuentran los siguientes archivos:
- `README.md`: Este archivo.
- `momento_1_model.py`: Script conteniendo la aplicacion de un modelo de machine learning y algunas predicciones (Este es el archivo a revisar).
- `metricas.py`: Script conteniendo algunas funciones para calcular metricas.
- `wine.data`: Dataset utilizado.
- `wine.names`: Descripción del dataset utilizado.

## Modelo de machine learning (Regresion logistica)
### Librerías utilizadas
- `pandas`: Para el manejo de datos.
- `numpy`: Para el manejo de datos.
- `sklearn`: Para realizar el split de entrenamiento y prueba.
- `math`: Para realizar operaciones matematicas.
- `os`: Para pulir las salidas a la consola.

### Descripción
La aplicacion del modelo de regresion logistica se encuentra en el archivo `momento_1_model.py`. En este
se puede encontrar la aplicación del modelo de manera manual, es decir, sin utilizar librerias de machine
learning.

### Predicciones
Se utilizaron los siguientes datos para realizar las predicciones los cuales son parte del dataset de prueba:

| Alcohol | Malic acid | Ash  | Alcalinity of ash | Magnesium | Total phenols | Flavanoids | Nonflavanoid phenols | Proanthocyanins | Color intensity | Hue  | OD280/OD315 of diluted wines | Proline |
|---------|------------|------|-------------------|-----------|---------------|------------|----------------------|-----------------|-----------------|------|------------------------------|---------|
| 13.64   | 3.1        | 2.56 | 15.2              | 116       | 2.7           | 3.03       | 0.17                 | 1.66            | 5.1             | 0.96 | 3.36                         | 845     |
| 14.21   | 4.04       | 2.44 | 18.9              | 111       | 2.85          | 2.65       | 0.3                  | 1.25            | 5.24            | 0.87 | 3.33                         | 1080    |
| 12.93   | 2.81       | 2.7  | 21.0              | 96        | 1.54          | 0.5        | 0.53                 | 0.75            | 4.6             | 0.77 | 2.31                         | 600     |
| 13.73   | 1.5        | 2.7  | 22.5              | 101       | 3.0           | 3.25       | 0.29                 | 2.38            | 5.7             | 1.19 | 2.71                         | 1285    |
| 12.37   | 1.17       | 1.92 | 19.6              | 78        | 2.11          | 2.0        | 0.27                 | 1.04            | 4.68            | 1.12 | 3.48                         | 510     |
| 14.3    | 1.92       | 2.72 | 20.0              | 120       | 2.8           | 3.14       | 0.33                 | 1.97            | 6.2             | 1.07 | 2.65                         | 1280    |
| 12.0    | 3.43       | 2.0  | 19.0              | 87        | 2.0           | 1.64       | 0.37                 | 1.87            | 1.28            | 0.93 | 3.05                         | 564     |
| 13.4    | 3.91       | 2.48 | 23.0              | 102       | 1.8           | 0.75       | 0.43                 | 1.41            | 7.3             | 0.7  | 1.56                         | 750     |
| 11.61   | 1.35       | 2.7  | 20.0              | 94        | 2.74          | 2.92       | 0.29                 | 2.49            | 2.65            | 0.96 | 3.26                         | 680     |
| 13.36   | 2.56       | 2.35 | 20.0              | 89        | 1.4           | 0.5        | 0.37                 | 0.64            | 5.6             | 0.7  | 2.47                         | 780     |

Se obtuvieron las siguientes predicciones:

| Prediccion obtenida | Prediccion esperada |
|---------------------|---------------------|
| 2                   | 1                   |
| 1                   | 1                   |
| 2                   | 3                   |
| 1                   | 1                   |
| 2                   | 2                   |
| 1                   | 1                   |
| 2                   | 2                   |
| 2                   | 3                   |
| 2                   | 2                   |
| 1                   | 3                   |





