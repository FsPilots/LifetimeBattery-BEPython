import pandas as pd
import cleandatatest as cd
import feature_eng_test as fe
import modeldatatest2_RL as RLmod
import feature_eng_test as fe
import modeleSVM as svm

# Charger le fichier CSV dans un DataFrame
dataframe = pd.read_csv('data/test_data/Battery_RUL_main_test.csv')

#Définir un chemin de sortie des données clean
output_clean_data_path='data/test_data/output_main/Battery_RUI_sortie_main_test.csv'

#Définir un chemin de sortie des données utiliables par l'IA
output_usable_data_path='data/test_data/output_main/Battery_RUI_usable_data.csv'

#Etape 1: Nettoyage des données
#cd.clean_data(dataframe,output_clean_data_path)

#Etape 2: Feature Engineering
#fe.feature_eng(output_clean_data_path,output_usable_data_path)

#Etape 3: Entrainement modèle IA et Analyse resultats
#RLmod.reglineaire(output_usable_data_path)

svm.modelregressionGPR(output_usable_data_path)

