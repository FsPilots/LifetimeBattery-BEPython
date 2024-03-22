import pandas as pd
import cleandata as cd
import feature_eng as fe
import modeldata_Reg_Linear as RLmod

# Charger le fichier CSV dans un DataFrame
dataframe = pd.read_csv('rendufinal/data/Battery_RUL.csv')

#Définir un chemin de sortie des données clean
output_clean_data_path='rendufinal/data/Battery_RUI_clean_data.csv'

#Définir un chemin de sortie des données utiliables par l'IA
output_usable_data_path='rendufinal/data/Battery_RUI_usable_data.csv'

#Etape 1: Nettoyage des données
cd.clean_data(dataframe,output_clean_data_path)

#Etape 2: Feature Engineering
fe.feature_eng(output_clean_data_path,output_usable_data_path)

#Etape 3: Entrainement modèle IA et Analyse resultats
RLmod.reglineaire(output_usable_data_path)

