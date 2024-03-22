import pandas as pd

def feature_eng(output_clean_data_path):
    # Import des bibliothèques nécessaires
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Chargement des données dans un DataFrame Pandas
# Assurez-vous de charger vos propres données ou d'utiliser un jeu de données de démonstration
df = pd.read_csv('data.csv')

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
plt.show()

# Analyse des composantes principales
# Vous pouvez examiner les composantes principales pour comprendre les relations entre les variables originales et les composantes
# Par exemple, pour afficher les poids des variables originales dans la première composante principale :
first_principal_component = pca.components_[0]
weights = pd.DataFrame({'Variable': X.columns,
                        'Weight': first_principal_component})
weights.sort_values(by='Weight', ascending=False, inplace=True)
print("Poids des variables dans la première composante principale:")
print(weights)

df




correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title('Heatmap de la matrice de corrélation')
plt.show()
