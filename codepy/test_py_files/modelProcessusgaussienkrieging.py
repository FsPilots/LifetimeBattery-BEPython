def modelregressionGPR(dataframe,output_clean_data_path,output_init_datastat_path,output_cleaned_datastat_path):

#Définir un chemin de sortie des statistiques finales
data='data/test_data/output_main/cleaned_data_stat_main_test.csv'

import numpy as np
import pandas as pd
import time
import GPy
import matplotlib.pyplot as plt

# Charger toutes les colonnes du fichier data.csv
data = pd.read_csv('data.csv')

# Extraire les valeurs des colonnes
X = data.iloc[:, :-1].values  # Toutes les colonnes sauf la dernière (cible)
Y = data.iloc[:, -1].values.reshape(-1, 1)  # Dernière colonne comme cible (Y)

# Diviser les données en ensemble d'entraînement et ensemble de test en utilisant 1/4 des données pour l'entraînement
split_index = len(X) // 4
X_train, X_test = X[:split_index], X[split_index:]
Y_train, Y_test = Y[:split_index], Y[split_index:]

# Créer le modèle de régression GPR avec un noyau linéaire
kernel = GPy.kern.Linear(input_dim=X_train.shape[1])
model = GPy.models.GPRegression(X_train, Y_train, kernel)

# Optimisation des hyperparamètres
model.optimize_restarts(num_restarts=5, verbose=True)

# Mesurer le temps d'entraînement
start_time = time.time()
model.optimize(messages=True)
training_time = time.time() - start_time

# Précision de la prédiction sur l'ensemble d'entraînement
mean_train, _ = model.predict(X_train)
train_accuracy = np.mean(np.abs(mean_train - Y_train))

# Précision de la prédiction sur l'ensemble de test
mean_test, _ = model.predict(X_test)
test_accuracy = np.mean(np.abs(mean_test - Y_test))

# Tracer les prédictions par rapport aux valeurs réelles pour l'ensemble d'entraînement
plt.figure(figsize=(10, 6))
plt.scatter(Y_train, mean_train, color='blue', label='Prédiction (Entraînement)')
plt.plot([Y_train.min(), Y_train.max()], [Y_train.min(), Y_train.max()], '--', color='gray', label='Valeurs réelles')
plt.xlabel('Valeurs réelles')
plt.ylabel('Prédictions')
plt.title('Prédictions vs Valeurs Réelles (Entraînement)')
plt.legend()
plt.grid(True)
plt.show()

# Tracer les prédictions par rapport aux valeurs réelles pour l'ensemble de test
plt.figure(figsize=(10, 6))
plt.scatter(Y_test, mean_test, color='red', label='Prédiction (Test)')
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], '--', color='gray', label='Valeurs réelles')
plt.xlabel('Valeurs réelles')
plt.ylabel('Prédictions')
plt.title('Prédictions vs Valeurs Réelles (Test)')
plt.legend()
plt.grid(True)
plt.show()

print(f"Temps d'entraînement : {training_time:.2f} secondes")
print(f"Précision de la prédiction sur l'ensemble d'entraînement : {train_accuracy:.4f}")
print(f"Précision de la prédiction sur l'ensemble de test : {test_accuracy:.4f}")

