import pandas as pd
import cleandatatest as cd

# Charger le fichier CSV dans un DataFrame
dataframe = pd.read_csv('data/test_data/Battery_RUL_test1.csv')

#Définir un chemin de sortie des données clean
outputcleandata='data/test_data/Battery_RUI_sortie_main_test.csv'

cd.clean_data(dataframe,outputcleandata)