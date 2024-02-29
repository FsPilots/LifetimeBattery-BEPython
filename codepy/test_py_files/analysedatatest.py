import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree

def analysedatatest(outputcleandata_path):
    # Chargement des données à partir d'un fichier CSV
    file_path = outputcleandata_path
    data = pd.read_csv(file_path)
    print("Colonnes du DataFrame :", data.columns)

    # Calculer la matrice de corrélation
    correlation_matrix = data.corr()

    # Afficher la matrice de corrélation sous forme de heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Matrice de corrélation")

    # Sélectionner les colonnes avec une corrélation supérieure à un certain seuil
    seuil_correlation = 0.8
    haute_correlation = (abs(correlation_matrix) > seuil_correlation)

    # Masquer les valeurs prépondérantes dans la matrice de corrélation
    correlation_matrix_masked = correlation_matrix.mask(haute_correlation)

    # Afficher la matrice de corrélation mise à jour sous forme de heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix_masked, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Matrice de corrélation avec les valeurs prépondérantes masquées")

    # Diviser les données en ensembles d'entraînement et de test
    X = data.drop(columns=['RUL'])  # Features
    y = data['RUL']  # Target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Construction du modèle d'arbre de régression
    model = DecisionTreeRegressor()

    # Entraînement du modèle
    model.fit(X_train, y_train)

    # Prédiction sur l'ensemble de test
    y_pred = model.predict(X_test)

    # Afficher l'arbre de régression
    plt.figure(figsize=(20, 20))
    plot_tree(model, filled=True, feature_names=X.columns, rounded=True, precision=2)
    #plt.savefig("regression_tree.png")  # Exportez l'arbre en tant qu'image PNG
    plt.show()

    # Calcul de l'erreur quadratique moyenne
    mse = mean_squared_error(y_test, y_pred)
    print("Erreur quadratique moyenne :", mse)