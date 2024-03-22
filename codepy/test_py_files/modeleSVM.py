def modelregressionGPR(dataframe,output_clean_data_path,output_init_datastat_path,output_cleaned_datastat_path):

#Définir un chemin de sortie des statistiques finales
data='data/test_data/output_main/cleaned_data_stat_main_test.csv'


import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
import time

# Charger toutes les colonnes du fichier data.csv
data = pd.read_csv('data.csv')

# Extraire les valeurs des colonnes
X = data.iloc[:, :-1].values  # Toutes les colonnes sauf la dernière (cible)
Y = data.iloc[:, -1].values.reshape(-1, 1)  # Dernière colonne comme cible (Y)

# Diviser les données en ensemble d'entraînement et ensemble de test en utilisant 1/4 des données pour l'entraînement
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

# Définition de la grille des hyperparamètres à optimiser
param_grid = {
    'C': [0.1, 1, 10, 100],  # Paramètre de régularisation
    'gamma': [0.01, 0.1, 1, 10],  # Paramètre du noyau (pour les noyaux rbf, poly et sigmoid)
}

# Créer le modèle de régression SVM
svm = SVR(kernel='rbf')

# Créer l'objet GridSearchCV pour l'optimisation
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5, n_jobs=-1)

# Effectuer la recherche sur grille pour trouver les meilleurs hyperparamètres
start_time = time.time()
grid_search.fit(X_train, Y_train.ravel())
training_time = time.time() - start_time

# Obtenir les meilleurs hyperparamètres
best_params = grid_search.best_params_
print("Meilleurs hyperparamètres trouvés :", best_params)

# Utiliser les meilleurs hyperparamètres pour créer le modèle final
best_svm = SVR(kernel='rbf', C=best_params['C'], gamma=best_params['gamma'])

# Entraîner le modèle final
best_svm.fit(X_train, Y_train.ravel())

# Mesurer le temps d'entraînement

# Précision de la prédiction sur l'ensemble d'entraînement
train_accuracy = best_svm.score(X_train, Y_train)

# Précision de la prédiction sur l'ensemble de test
test_accuracy = best_svm.score(X_test, Y_test)

print(f"Temps d'entraînement : {training_time:.2f} secondes")
print(f"Précision de la prédiction sur l'ensemble d'entraînement : {train_accuracy:.4f}")
print(f"Précision de la prédiction sur l'ensemble de test : {test_accuracy:.4f}")
