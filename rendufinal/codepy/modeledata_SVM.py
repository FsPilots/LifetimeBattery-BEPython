import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt

def modelregressionGPR(data_path):
    print('.....Modele IA SVM.....')
    # Commencer le chronomètre
    debut = time.time()
    # Charger toutes les colonnes du fichier data.csv
    data = pd.read_csv(data_path)

    # Extraire les valeurs des colonnes
    X = data.iloc[:, :-1].values  # Toutes les colonnes sauf la dernière (cible)
    Y = data.iloc[:, -1].values.reshape(-1, 1)  # Dernière colonne comme cible (Y)

    # Diviser les données en ensemble d'entraînement et ensemble de test en utilisant 1/4 des données pour l'entraînement
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

    # Utiliser les meilleurs hyperparamètres trouvés
    best_params = {'C': 1000, 'gamma': 0.000001}

    # Créer le modèle final avec les meilleurs hyperparamètres et la régularisation
    best_svm = SVR(kernel='rbf', C=best_params['C'], gamma=best_params['gamma'])

    # Entraîner le modèle final avec régularisation
    start_time = time.time()
    best_svm.fit(X_train, Y_train.ravel())
    training_time = time.time() - start_time

    # Prédictions sur l'ensemble d'entraînement
    train_predictions = best_svm.predict(X_train)

    # Prédictions sur l'ensemble de test
    test_predictions = best_svm.predict(X_test)

    # Tracer les prédictions par rapport aux vraies valeurs
    plt.figure(figsize=(10, 5))

    # Plot pour l'ensemble d'entraînement
    plt.subplot(1, 2, 1)
    plt.scatter(Y_train, train_predictions, color='blue')
    plt.plot([min(Y_train), max(Y_train)], [min(Y_train), max(Y_train)], color='red', linestyle='--')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Training Set Performance')

    # Plot pour l'ensemble de test
    plt.subplot(1, 2, 2)
    plt.scatter(Y_test, test_predictions, color='blue')
    plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], color='red', linestyle='--')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Test Set Performance')

    plt.tight_layout()
    plt.show()

    # Mesurer le temps d'entraînement
    print(f"Temps d'entrainement : {training_time:.2f} secondes")

    # Précision de la prédiction sur l'ensemble d'entraînement
    train_accuracy = best_svm.score(X_train, Y_train)
    print(f"Precision de la prediction sur l'ensemble d'entrainement : {train_accuracy:.4f}")

    # Précision de la prédiction sur l'ensemble de test
    test_accuracy = best_svm.score(X_test, Y_test)
    print(f"Precision de la prediction sur l'ensemble de test : {test_accuracy:.4f}")
    
