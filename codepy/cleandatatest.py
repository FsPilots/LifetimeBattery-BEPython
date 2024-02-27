import pandas as pd

# Charger le fichier CSV dans un DataFrame
dataframe = pd.read_csv('data/Battery_RUL_test1.csv')

# Afficher les premières lignes du DataFrame
print(dataframe.head())
