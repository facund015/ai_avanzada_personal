"""
Title: Momento Retro 2 Model
Author: Facundo Vecchi A01283666
Date: 2022-09-07
Description: En este programa se aplican varios modelos de clasificacion utilizando librerias varias
"""

# Uso General
import pandas as pd

# Metricas y utilidades
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

# Modelos
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings('ignore')
from xgboost import XGBClassifier

"""
Varias librerias me estaban dando warnings, asi que los deshabilite.
Todos los warnings que me aparecian eran de librerias que utilizaban
metodos que estan deprecados, pero que aun funcionan (la importacion
de xgboost me un warning como este).

La unica excepcion es el warning de MLPClassifier, que me dice que
no se alcanzo la convergencia, pero el modelo funciona igual, y
finalmente no es el que utilizare para las predicciones.
"""

# Importacion de Datos
print("-" * 100)
print("Importando datos...")

cols = ["Class", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols", "Flavanoids",
        "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"]
data = pd.read_csv("wine.data", header=None, names=cols)

# Separacion de datos y reduccion de dimensionalidad
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(data.drop(columns="Class", axis=1)))
y = data.Class.values
pca = PCA(n_components=2, random_state=42)
pca.fit(X)

# Separacion de datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

print("Datos importados correctamente.")

# Aplicacion de modelos
print("Entrenando modelos...")

models_normal = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(),
    "SVM": SVC(random_state=42),
    "MLP": MLPClassifier(random_state=42),
    "XGBoost": XGBClassifier(verbosity=0, silent=True),
    "LightGBM": LGBMClassifier(random_state=42)
}
results_normal = []

for name, model in models_normal.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results_normal.append(accuracy_score(y_test, y_pred) * 100)
    # print(f"{name}, Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

models_pca = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(),
    "SVM": SVC(random_state=42),
    "MLP": MLPClassifier(random_state=42),
    "XGBoost": XGBClassifier(verbosity=0, silent=True),
    "LightGBM": LGBMClassifier(random_state=42)
}
results_pca = []

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

for name, model in models_pca.items():
    model.fit(X_train_pca, y_train)
    y_pred = model.predict(X_test_pca)
    results_pca.append(accuracy_score(y_test, y_pred) * 100)
    # print(f"{name}, Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

print("Modelos entrenados correctamente.")
print("-" * 100)

# Resultados
temp = len("Resultados")
temp2 = int(temp / 2)
print("-" * (49 - temp2), "Resultados", "-" * (49 - temp2))

model_names = list(models_normal.keys())

res = pd.DataFrame(columns=["Model", "Accuracy_normal", "Accuracy_pca"])
res.Model = model_names
res.Accuracy_normal = [f"{x:.2f} %" for x in results_normal]
res.Accuracy_pca = [f"{x:.2f} %" for x in results_pca]
print(res)

# Conclusiones
temp = len("Conclusiones")
temp2 = int(temp / 2)
print("-" * (49 - temp2), "Conclusiones", "-" * (49 - temp2))
print("Se entreno un total de 16 modelos, 8 con los datos originales escalados y")
print("8 con los datos reducidos a 2 dimensiones utilizando el metodo de PCA.\n")

print("Se puede observar que al aplicar PCA, los modelos de Decision Tree,")
print("KNN y XGBoost obtuvieron una mejora en su accuracy, mientras que los")
print("modelos de Random Forest, Logistic Regression, SVM, MLP y LightGBM")
print("obtuvieron una disminucion o ningun cambio en su accuracy.\n")

print("En este caso, utilizaremos tan solo el modelo de Random Forest de Sklearn")
print("para realizar las predicciones, ya que tuvo un accuracy de 100% en el primer caso")

print("-" * 100)
print("Resultados a detalle del Random Forest entrenada con los datos originales escalados:")
print(classification_report(y_test, models_normal["Random Forest"].predict(X_test)))
print("Matriz de confusion:")
print(confusion_matrix(y_test, models_normal["Random Forest"].predict(X_test)))
print("-" * 100)
print("Resultados a detalle del Random Forest entrenada con los datos reducidos a 2 dimensiones:")
print(classification_report(y_test, models_pca["Random Forest"].predict(X_test_pca)))
print("Matriz de confusion:")
print(confusion_matrix(y_test, models_pca["Random Forest"].predict(X_test_pca)))

# Predicciones
temp = len('Predicciones')
temp2 = int(temp / 2)
print("-" * (49 - temp2), 'Predicciones', "-" * (49 - temp2))

cases = X_test[:10]
ypreds_normal = models_normal["Random Forest"].predict(X_test[:10])
ypreds_pca = models_pca["Random Forest"].predict(X_test_pca[:10])
ytrues = y_test[:10]
columns = cols[1:]

# pd.DataFrame(scaler.inverse_transform(cases)).to_csv("cases.csv", index=False, header=columns)
#
# results = pd.DataFrame(columns=["Normal", "PCA", "True"])
# results.Normal = ypreds_normal
# results.PCA = ypreds_pca
# results["True"] = ytrues
# results.to_csv("results.csv", index=False)

for i in range(len(cases)):
    print(f"Prediccion {i + 1}:")
    print("Datos entrada:")
    case = pd.DataFrame(cases.iloc[i]).T
    case.columns = columns
    print(case.to_string(index=False))
    print(f"\nPrediccion normal: {ypreds_normal[i]}, Prediccion con PCA: {ypreds_pca[i]}, Valor real: {ytrues[i]}")
    print("-" * 100)


