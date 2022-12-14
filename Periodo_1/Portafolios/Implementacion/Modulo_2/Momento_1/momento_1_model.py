import numpy as np
import pandas as pd
import math as m
import metricas as mt
from sklearn.model_selection import train_test_split
import os


def predict_all_classes(X_, thetas):
    p = []
    for k in range(n_wine_class):
        p.append(
            h(X_.x, thetas[k], X_.x1, X_.x2, X_.x3, X_.x4, X_.x5, X_.x6, X_.x7, X_.x8, X_.x9, X_.x10, X_.x11, X_.x12))
    return p.index(max(p)) + 1


def h(x, theta, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12):
    return 1 / (1 + m.exp(-(
            theta[0] + theta[1] * x + theta[2] * x1 + theta[3] * x2 + theta[4] * x3 + theta[5] * x4 + theta[
        6] * x5 + theta[7] * x6 + theta[8] * x7 + theta[9] * x8 + theta[10] * x9 + theta[11] * x10 + theta[
                12] * x11 + theta[13] * x12)))


temp = len("Momento de Retroalimentación 1: Módulo 2")
temp2 = int(temp/2)
print("-" * (49-temp2), "Momento de Retroalimentación 1: Módulo 2", "-" * (49-temp2))
print("Facundo Vecchi A01283666")
print("Base de datos siendo utilizada: Wine")
print("Columnas utilizadas para el entrenamiento del modelo: 13")
print("El modelo utilizado es regresión logística")
print("-" * 100)

cols = ["Class", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols", "Flavanoids",
        "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"]
df = pd.read_csv("wine.data", header=None, names=cols)
wine_classes = df["Class"].unique().tolist()
n_wine_class = len(wine_classes)

X_ = df.drop(columns=["Class"], axis=1)
X_.columns = ["x", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12"]
y_ = df.Class.values

np.random.seed(42)
X_train, X_test, y_train, y_test = train_test_split(X_, y_)

alpha = 0.00005
iters = 300
thetas = np.full((n_wine_class, 14), 0.00001)

n_train = len(y_train)

# print(X_test.iloc[:10])
# print(pd.DataFrame(X_test.iloc[19]).T.to_string(index=False))
#
# temp_df = X_test.iloc[:10]
# temp_df.columns = cols[1:]
# temp_df.to_csv("test.csv", index=False)
# temp_df_y = y_test[:10]
# temp_df_y = pd.DataFrame(temp_df_y, columns=["Class"])
# temp_df_y.to_csv("test_y.csv", index=False)

print("Entrenando el modelo... (Este proceso puede tardar unos minutos)")
for k in range(n_wine_class):
    for idx in range(iters):
        acumDelta = {"x_": [], "x0": [],
                     "x1": [], "x2": [],
                     "x3": [], "x4": [],
                     "x5": [], "x6": [],
                     "x7": [], "x8": [],
                     "x9": [], "x10": [],
                     "x11": [], "x12": []}
        sJt = {"x_": [], "x0": [],
               "x1": [], "x2": [],
               "x3": [], "x4": [],
               "x5": [], "x6": [],
               "x7": [], "x8": [],
               "x9": [], "x10": [],
               "x11": [], "x12": []}
        for (i_row, X), y in zip(X_train.iterrows(), y_train):
            if y != (k + 1):
                # print(k+1, y, 'replaced')
                y = 0
            else:
                # print(k+1, y)
                y = 1

            acumDelta['x_'].append(
                h(X.x, thetas[k], X.x1, X.x2, X.x3, X.x4, X.x5, X.x6, X.x7, X.x8, X.x9, X.x10, X.x11, X.x12) - y)

            for i in range(13):
                acumDelta['x' + str(i)].append(
                    (h(X.x, thetas[k], X.x1, X.x2, X.x3, X.x4, X.x5, X.x6, X.x7, X.x8, X.x9, X.x10, X.x11, X.x12) - y) *
                    X.iloc[i])

        sJt['x_'] = sum(acumDelta['x_'])
        for i in range(13):
            sJt['x' + str(i)] = sum(acumDelta['x' + str(i)])

        thetas[k][0] = thetas[k][0] - alpha / n_train * sJt['x_']
        for i in range(1, 14):
            thetas[k][i] = thetas[k][i] - (alpha / n_train) * sJt['x' + str(i - 1)]

print("Modelo entrenado")
temp = len("Thetas")
temp2 = int(temp/2)
print("-" * (49-temp2), "Thetas", "-" * (49-temp2))
print(thetas)
temp = len("Metricas de evaluacion")
temp2 = int(temp/2)
print("-" * (49-temp2), "Metricas de evaluacion", "-" * (49-temp2))

predicts = []
for idx, value in X_test.iterrows():
    predicts.append(predict_all_classes(value, thetas))

acc, hits, misses = mt.accuracy_simple(predicts, y_test)
print('Precision:', acc, '%')
print('Acertadas:', hits)
print('Fallas:', misses)

temp = len("Predicciones")
temp2 = int(temp/2)
print("-" * (49-temp2), "Predicciones", "-" * (49-temp2))

for idx, value in X_test.iloc[:10].reset_index(drop=True).iterrows():
    print("\nPredicción para la muestra", idx + 1)
    print("----------------------------")
    print('Entradas:')
    print(pd.DataFrame(value).T.to_string(index=False))
    print('Prediccion:', predict_all_classes(value, thetas), 'Esperado:', y_test[idx])

os.system('pause')
