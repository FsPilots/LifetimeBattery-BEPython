import numpy as np
import pandas as pd
import GPy 
import time


def modelregressionGPR(data_path):
    print('.....Modele IA Processus Gaussien.....')
    
    # Charger toutes les colonnes du fichier data.csv
    data = pd.read_csv(data_path)

    # Extraire les valeurs des colonnes
    X = data.iloc[:, :-1].values  # Toutes les colonnes sauf la dernière (cible)
    Y = data.iloc[:, -1].values.reshape(-1, 1)  # Dernière colonne comme cible (Y)

    # Diviser les données en ensemble d'entraînement et ensemble de test en utilisant 1/4 des données pour l'entraînement
    split_index = len(X) // 4
    X_train, X_test = X[:split_index], X[split_index:]
    Y_train, Y_test = Y[:split_index], Y[split_index:]

    # Créer le modèle de régression GPR avec kriging
    kernel = GPy.kern.RBF(input_dim=X_train.shape[1], variance=1., lengthscale=1.)
    model = GPy.models.GPRegression(X_train, Y_train, kernel)

    # Supprimer les contraintes existantes des paramètres du modèle
    model.unconstrain()
    # Ajouter une régularisation L2 (Ridge) au modèle
    alpha = 0.1
    model.rbf.variance.set_prior(GPy.priors.Gaussian(0., alpha))
    model.rbf.lengthscale.set_prior(GPy.priors.Gaussian(0., alpha))
    model.Gaussian_noise.variance.set_prior(GPy.priors.Gaussian(0., alpha))

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

    print(f"Temps d'entrainement : {training_time:.2f} secondes")
    print(f"Precision de la prediction sur l'ensemble d'entrainement : {train_accuracy:.4f}")
    print(f"Precision de la prediction sur l'ensemble de test : {test_accuracy:.4f}")

    return(round(test_accuracy,4),round(training_time,4))


