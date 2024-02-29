import pandas as pd
import cleandatatest as cd
import analysedatatest as adt

# Charger le fichier CSV dans un DataFrame
dataframe = pd.read_csv('data/test_data/Battery_RUL_main_test.csv')

#Définir un chemin de sortie des données clean
output_clean_data_path='data/test_data/output_main/Battery_RUI_sortie_main_test.csv'

#Définir un chemin de sortie des statistiques initiales
output_init_datastat_path='data/test_data/output_main/init_stat_main_test.csv'

#Définir un chemin de sortie des statistiques finales
output_cleaned_datastat_path='data/test_data/output_main/cleaned_data_stat_main_test.csv'

cd.clean_data(dataframe,output_clean_data_path,output_init_datastat_path,output_cleaned_datastat_path)
#adt.analysedatatest(output_clean_data_path)