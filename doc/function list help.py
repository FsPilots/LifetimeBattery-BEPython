import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, roc_curve, auc
from sklearn.preprocessing import LabelBinarizer

# Charger les données

data = pd.read_csv("C:/Users/thoma/Documents/GitHub/LifetimeBattery-BEPython/data/Battery_RUL.csv")

# Supprimer les données aberrantes en utilisant la méthode de l'écart interquartile
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]

# REGRESSION :

# Diviser les données en caractéristiques (features) et cible (target)
X = data[['Cycle_Index', 'Discharge Time (s)', 'Decrement 3.6-3.4V (s)', 'Max. Voltage Dischar. (V)', 'Min. Voltage Charg. (V)', 'Time at 4.15V (s)', 'Time constant current (s)', 'Charging time (s)']]
y = data['RUL']

# Diviser les données en ensemble d'entraînement et ensemble de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialiser et entraîner le modèle de régression linéaire
model = LinearRegression()
model.fit(X_train, y_train)

# Faire des prédictions sur l'ensemble de test
predictions = model.predict(X_test)

# Calculer l'erreur quadratique moyenne (RMSE)
rmse = mean_squared_error(y_test, predictions, squared=False)
print("Root Mean Squared Error (RMSE):", rmse)


# CLASSIFICARION :


# Convertir la variable cible (RUL) en classes
# Par exemple, diviser en 3 classes (courte durée de vie, moyenne durée de vie, longue durée de vie)
data['RUL_Class'] = pd.cut(data['RUL'], bins=3, labels=['Courte durée de vie', 'Moyenne durée de vie', 'Longue durée de vie'])

# Diviser les données en caractéristiques (features) et cible (target)
X = data[['Cycle_Index', 'Discharge Time (s)', 'Decrement 3.6-3.4V (s)', 'Max. Voltage Dischar. (V)', 'Min. Voltage Charg. (V)', 'Time at 4.15V (s)', 'Time constant current (s)', 'Charging time (s)']]
y = data['RUL_Class']

# Diviser les données en ensemble d'entraînement et ensemble de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialiser et entraîner le modèle de classification RandomForest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Faire des prédictions sur l'ensemble de test
predictions = model.predict(X_test)

# Calculer l'exactitude (accuracy) du modèle
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# Afficher le rapport de classification
print("Classification Report:")
print(classification_report(y_test, predictions))

# COURBES :

# Tracer des scatter plots pour chaque caractéristique par rapport à la variable cible (RUL)
features = ['Cycle_Index', 'Discharge Time (s)', 'Decrement 3.6-3.4V (s)', 'Max. Voltage Dischar. (V)', 'Min. Voltage Charg. (V)', 'Time at 4.15V (s)', 'Time constant current (s)', 'Charging time (s)']

plt.figure(figsize=(15, 10))
for i, feature in enumerate(features):
    plt.subplot(3, 3, i+1)
    plt.scatter(data[feature], data['RUL'], alpha=0.5)
    plt.xlabel(feature)
    plt.ylabel('RUL')
    plt.title(f'{feature} vs RUL')

plt.tight_layout()
plt.show()

# SYSTEME DE POIDS :

def predict_rul(Cycle_Index, Discharge_Time, Decrement_Time, Max_Voltage_Discharge, Min_Voltage_Charge, Time_at_4_15V,
                Time_Constant_Current, Charging_Time):

    # Définir les coefficients de pondération pour chaque paramètre (à ajuster en fonction de l'impact perçu)
    weights = {
        'Cycle_Index': 0.1,
        'Discharge_Time': 0.2,
        'Decrement_Time': 0.15,
        'Max_Voltage_Discharge': 0.1,
        'Min_Voltage_Charge': 0.1,
        'Time_at_4_15V': 0.15,
        'Time_Constant_Current': 0.05,
        'Charging_Time': 0.15
    }

    # Calculer le score en combinant les paramètres pondérés
    score = (Cycle_Index * weights['Cycle_Index'] +
             Discharge_Time * weights['Discharge_Time'] +
             Decrement_Time * weights['Decrement_Time'] +
             Max_Voltage_Discharge * weights['Max_Voltage_Discharge'] +
             Min_Voltage_Charge * weights['Min_Voltage_Charge'] +
             Time_at_4_15V * weights['Time_at_4_15V'] +
             Time_Constant_Current * weights['Time_Constant_Current'] +
             Charging_Time * weights['Charging_Time'])

    return score


# Exemple d'utilisation de la fonction pour prédire la durée de vie résiduelle
predicted_rul = predict_rul(100, 200, 300, 4.2, 3.5, 50, 10, 500)
print("Predicted RUL:", predicted_rul)

# SYSTEME DE POIDS DEVELOPPES :
#Dans ce code, la fonction learn_weights entraîne un modèle de régression linéaire sur les données d'entraînement pour estimer les poids optimaux pour chaque paramètre. Ensuite, la fonction predict_rul utilise les poids appris pour prédire la durée de vie résiduelle en combinant les paramètres pondérés. Cette approche permet d'apprendre les poids à partir des données, ce qui peut améliorer les performances du modèle en utilisant une régression linéaire pour estimer les poids optimaux.

def learn_weights(X, y):

    # Diviser les données en ensemble d'entraînement et ensemble de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialiser et entraîner le modèle de régression linéaire
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Obtenir les poids appris à partir du modèle
    weights = model.coef_

    return weights

def predict_rul(X, weights):
    # Calculer le score en combinant les paramètres pondérés
    score = X.dot(weights)

    return score


# Charger les données
data = pd.read_csv("C:/Users/thoma/Documents/GitHub/LifetimeBattery-BEPython/data/Battery_RUL.csv")  # Assure-toi de remplacer "ton_fichier.csv" par le nom de ton fichier de données

# Identifier les valeurs aberrantes dans chaque colonne
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1

# Définir un seuil pour détecter les valeurs aberrantes (par exemple, 1.5 fois l'IQR)
threshold = 1.5
lower_bound = Q1 - threshold * IQR
upper_bound = Q3 + threshold * IQR

# Supprimer les lignes contenant des valeurs aberrantes dans au moins une colonne
data_cleaned = data[~((data < lower_bound) | (data > upper_bound)).any(axis=1)]

# Afficher les lignes supprimées
outliers = data[~data.index.isin(data_cleaned.index)]
print("Valeurs aberrantes supprimées:")
print(outliers)

# Diviser les données en caractéristiques (features) et cible (target)
X = data[
    ['Cycle_Index', 'Discharge Time (s)', 'Decrement 3.6-3.4V (s)', 'Max. Voltage Dischar. (V)', 'Min. Voltage Charg. (V)', 'Time at 4.15V (s)', 'Time constant current (s)', 'Charging time (s)']]
y = data['RUL']

# Apprendre les poids à partir des données
weights = learn_weights(X, y)

# Prédire la durée de vie résiduelle en utilisant les poids appris
predicted_rul = predict_rul(X, weights)
print("Predicted RUL:", predicted_rul)