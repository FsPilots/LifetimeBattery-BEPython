import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor, plot_tree

def abre_de_regression(outputcleandata_path):

    # Chargement des données à partir d'un fichier CSV
    data = pd.read_csv(outputcleandata_path)
    print("Colonnes du DataFrame :", data.columns)

    # Diviser les données en ensembles d'entraînement et de test
    X = data.drop(columns=['RUL'])  # Features
    y = data['RUL']  # Target variable

    # Utiliser train_test_split pour diviser les données
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Affichage des dimensions des ensembles d'entraînement et de test
    print("Taille de l'ensemble d'entraînement X :", X_train.shape)
    print("Taille de l'ensemble de test X :", X_test.shape)
    print("Taille de l'ensemble d'entraînement y :", y_train.shape)
    print("Taille de l'ensemble de test y :", y_test.shape)

    # Définir la grille de paramètres à explorer
    param_grid = {
        'max_depth': [3, 5, 7, 9, 11]  # Vous pouvez ajuster cette liste selon vos besoins
    }

    # Créer le modèle d'arbre de régression
    tree_reg = DecisionTreeRegressor()

    # Effectuer une recherche de grille avec validation croisée
    grid_search = GridSearchCV(estimator=tree_reg, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error',
                               verbose=1)
    grid_search.fit(X_train, y_train)

    # Afficher les meilleurs paramètres trouvés
    print("Meilleurs paramètres :", grid_search.best_params_)

    # Utiliser le meilleur modèle pour prédire sur l'ensemble de test
    best_model = grid_search.best_estimator_

    # Prédiction sur l'ensemble de test
    y_pred = best_model.predict(X_test)

    # Calcul de l'erreur quadratique moyenne
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Erreur quadratique moyenne:", mse)
    print("R^2:", r2)

    # Vous pouvez également visualiser l'arbre de régression si vous le souhaitez
    plt.figure(figsize=(20, 20))
    plot_tree(best_model, filled=True, feature_names=X.columns, rounded=True, precision=2)
    plt.show()
    plt.savefig("regression_tree.png")  # Exportez l'arbre en tant qu'image PNG

    #PERFORMANCES
    depths = [3, 5, 7, 9, 11]

    # Initialiser des listes pour enregistrer les performances
    mse_scores = []
    r2_scores = []

    # Pour chaque profondeur, effectuer une recherche de grille et enregistrer les performances
    for depth in depths:
        # Créer le modèle d'arbre de régression
        tree_reg = DecisionTreeRegressor(max_depth=depth)

        # Entraînement du modèle
        tree_reg.fit(X_train, y_train)

        # Prédiction sur l'ensemble de test
        y_pred = tree_reg.predict(X_test)

        # Calcul des performances
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Enregistrer les performances
        mse_scores.append(mse)
        r2_scores.append(r2)

    # Tracer les performances
    plt.figure(figsize=(10, 5))
    plt.plot(depths, mse_scores, label='MSE', marker='o')
    plt.xlabel('Profondeur de l\'arbre MSE')
    plt.ylabel('Performance')
    plt.title('Performance du modèle en fonction de la profondeur de l\'arbre')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(depths, r2_scores, label='R²', marker='o')
    plt.xlabel('Profondeur de l\'arbre R²')
    plt.ylabel('Performance')
    plt.title('Performance du modèle en fonction de la profondeur de l\'arbre')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Initialiser des listes pour enregistrer les performances
    train_errors = []
    test_errors = []

    # Pour chaque profondeur, entraîner le modèle et enregistrer les erreurs d'entraînement et de test
    for depth in depths:
        # Créer le modèle d'arbre de régression
        tree_reg = DecisionTreeRegressor(max_depth=depth)

        # Entraîner le modèle
        tree_reg.fit(X_train, y_train)

        # Prédiction sur l'ensemble d'entraînement
        y_train_pred = tree_reg.predict(X_train)
        train_error = mean_squared_error(y_train, y_train_pred)
        train_errors.append(train_error)

        # Prédiction sur l'ensemble de test
        y_test_pred = tree_reg.predict(X_test)
        test_error = mean_squared_error(y_test, y_test_pred)
        test_errors.append(test_error)

    # Tracer les courbes d'apprentissage
    plt.figure(figsize=(10, 5))
    plt.plot(depths, train_errors, label='Entraînement', marker='o')
    plt.plot(depths, test_errors, label='Test', marker='o')
    plt.xlabel('Profondeur de l\'arbre')
    plt.ylabel('Erreur quadratique moyenne (MSE)')
    plt.title('Courbes d\'apprentissage')
    plt.legend()
    plt.grid(True)
    plt.show()

#RUN
output_clean_data_path = 'C:/Users/thoma/Documents/GitHub/LifetimeBattery-BEPython/data/test_data/output_main/Battery_RUI_sortie_main_test.csv'
abre_de_regression(output_clean_data_path)