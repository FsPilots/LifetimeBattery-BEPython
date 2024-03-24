import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def reglineaire(data_path):
    print('.....Modele IA Regression Lineaire.....')
    # Charger les données à partir du fichier CSV
    data = pd.read_csv(data_path)

    # Séparer les caractéristiques (X) et les étiquettes (y)
    X = data.drop('RUL', axis=1)  # Supprimer la colonne RUL pour obtenir les caractéristiques
    y = data['RUL']  # Définir les étiquettes comme la colonne RUL

    #print('X:',X)
    #print('Y:',y)

    # Fractionner les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Créer et entraîner le modèle de régression linéaire
    start_time = time.time()
    model = LinearRegression()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    # Faire des prédictions sur l'ensemble de test
    predictions = model.predict(X_test)

    # Mesurer le temps d'entraînement
    print(f"Temps d'entrainement : {training_time:.2f} secondes")
    
    # Évaluer la performance du modèle
    mse = mean_squared_error(y_test, predictions)
    test_accuracy = r2_score(y_test, predictions)
    print("Erreur quadratique moyenne :", mse)
    print("Precision de la prediction sur l'ensemble de test' (Coef R2) :", test_accuracy)

    # Créer une nouvelle figure avec deux sous-graphiques
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Tracer les résultats de l'ensemble d'entraînement
    axes[0].scatter(y_train, model.predict(X_train), color='blue', label='Entraînement')
    axes[0].plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], '--', color='gray')  # Ligne de référence (y = x)
    axes[0].set_title('Prédiction vs Réalité - Ensemble d\'entraînement')
    axes[0].set_xlabel('RUL Réel')
    axes[0].set_ylabel('RUL Prédit')
    axes[0].legend()
    axes[0].grid(True)

    # Tracer les résultats de l'ensemble de test
    axes[1].scatter(y_test, predictions, color='red', label='Test')
    axes[1].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='gray')  # Ligne de référence (y = x)
    axes[1].set_title('Prédiction vs Réalité - Ensemble de test')
    axes[1].set_xlabel('RUL Réel')
    axes[1].set_ylabel('RUL Prédit')
    axes[1].legend()
    axes[1].grid(True)
        
    # Afficher la figure
    plt.tight_layout()
    plt.savefig('rendufinal/doc/resultats/figure_reg_lineaire.png')
    #plt.show()
    
    return(round(test_accuracy,4),round(training_time,4))
