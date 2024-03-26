import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def feature_eng(clean_data_path,usable_data_path):
    print('\n.....Feature Engineering / Analyses.....')

    # Chargement des données dans un DataFrame Pandas
    # Assurez-vous de charger vos propres données ou d'utiliser un jeu de données de démonstration
    df = pd.read_csv(clean_data_path)

    # Sélection des caractéristiques pour l'analyse PCA
    X = df.iloc[:, :-1:]  # Exclut la derniere colonne
    y = df.iloc[:, -1]

    # Standardisation des données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Application de l'analyse en composantes principales (PCA)
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    # Variance expliquée par chaque composante principale
    explained_variance_ratio = pca.explained_variance_ratio_

    # Affichage du pourcentage de variance expliquée par chaque composante
    print("Pourcentage de variance expliquée par chaque composante:")
    print(explained_variance_ratio)

    # Tracé du graphique du pourcentage de variance expliquée par chaque composante
    plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o')
    plt.xlabel('Composante principale')
    plt.ylabel('Pourcentage de variance expliquée')
    plt.title('Pourcentage de variance expliquée par composante principale')
    plt.savefig('rendufinal/doc/resultats/pourcentage_variance.png')
    #plt.show()
    

    # Analyse des composantes principales
    # Vous pouvez examiner les composantes principales pour comprendre les relations entre les variables originales et les composantes
    # Par exemple, pour afficher les poids des variables originales dans la première composante principale :
    first_principal_component = pca.components_[0]
    weights = pd.DataFrame({'Variable': X.columns,
                            'Weight': first_principal_component})
    weights.sort_values(by='Weight', ascending=False, inplace=True)
    print("Poids des variables dans la premiere composante principale:")
    print(weights)

    correlation_matrix = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
    plt.title('Heatmap de la matrice de correlation')
    plt.savefig('rendufinal/doc/resultats/matrice_corr.png')
    #plt.show()
    

    # Recherche des colonnes avec une corrélation supérieure à 0.99
    redundant_columns = set()
    for i in range(len(correlation_matrix.columns) - 1):  # Ignorer la dernière colonne
        for j in range(i+1, len(correlation_matrix.columns) - 1):  # Commence à i+1 pour éviter de considérer les valeurs de la diagonale
            if abs(correlation_matrix.iloc[i, j]) > 0.98:
                col_i = correlation_matrix.columns[i]
                col_j = correlation_matrix.columns[j]
                redundant_columns.add((col_i, col_j))  # Ajoute un tuple pour garder une trace des paires de colonnes redondantes

    # Affichage des colonnes redondantes
    print("Colonnes redondantes :")
    print(redundant_columns)

    # Suppression des colonnes redondantes
    for col_pair in redundant_columns:
        # On supprime seulement une des colonnes de chaque paire, en excluant la première colonne du tableau
        if col_pair[1] in df.columns[1:]:
            df.drop(col_pair[1], axis=1, inplace=True)

    # Affichage du DataFrame après suppression des colonnes redondantes
    print("DataFrame apres suppression des colonnes redondantes :")
    print(df.head())
    print (df.shape)
    df.to_csv(usable_data_path)
    
    
    
