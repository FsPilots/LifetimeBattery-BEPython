import pandas as pd

# Charger le fichier CSV dans un DataFrame
dataframe = pd.read_csv('data/test_data/Battery_RUL_test1.csv')

#Définir un chemin de sortie des données clean
output_clean_data_path='data/test_data/output_test1/Battery_RUItest1_sortie.csv'

#Définir un chemin de sortie des statistiques initiales
output_init_datastat_path='data/test_data/output_test1/init_stat1.csv'

#Définir un chemin de sortie des statistiques finales
output_cleaned_datastat_path='data/test_data/output_test1/cleaned_data_stat1.csv'

#Tri des données
def clean_data(dataframe,output_clean_data_path,output_init_datastat_path,output_cleaned_datastat_path):
    
    #Affiche les premières valeurs pour vérifier la bonne lecture du fichier
    print(dataframe.head())
    
    #Affiche le nb de données
    print('Taille initiale:',dataframe.shape)
    
    #Suppression des lignes avec des valeurs manquantes
    dataframe=dataframe.dropna(axis=0)
    
    #Affiche le nb de données après effactement des lignes vides
    print('Taille after supr na:',dataframe.shape)
    
    #Affiche les premières statistiques
    init_stat=dataframe.describe()
    print(init_stat)
    init_stat.to_csv(output_init_datastat_path)
    
    #Nettoyage
    
    # Suppression des lignes contenant des valeurs négatives dans n'importe quelle colonne
    dataframe = dataframe.loc[(dataframe >= 0).all(axis=1)]
    
    #Affiche le nb de données après effactement des lignes avec valeurs négatives
    print('Taille after supr neg:',dataframe.shape)
    
    #Suppression des lignes avec des erreurs (basées sur la vérification par la valeur moyenne x 2 en max)
    #Concerne les colonnes Discharge Time (s)//Decrement 3.6-3.4V (s)//Time at 4.15V (s)//Time constant current (s)//Charging time (s)

    # Charger le fichier CSV init_stat
    datastat = pd.read_csv(output_init_datastat_path)
    
    #Affiche les premières valeurs pour vérifier la bonne lecture du fichier
    #print(datastat.head())
    
    print('Moyenne des parametres initiaux:')
    print('Discharge Time (s):',datastat.loc[1,'Discharge Time (s)'])
    print('Decrement 3.6-3.4V (s):',datastat.loc[1,'Decrement 3.6-3.4V (s)'])
    print('Time at 4.15V (s):',datastat.loc[1,'Time at 4.15V (s)'])
    print('Time constant current (s):',datastat.loc[1,'Time constant current (s)'])
    print('Charging time (s):',datastat.loc[1,'Charging time (s)'])

    #Nettoyage du dataframe
    dataframe=dataframe.loc[dataframe['Discharge Time (s)']<(datastat.loc[1,'Discharge Time (s)'])*2]
    dataframe=dataframe.loc[dataframe['Decrement 3.6-3.4V (s)']<(datastat.loc[1,'Decrement 3.6-3.4V (s)'])*2]
    dataframe=dataframe.loc[dataframe['Time at 4.15V (s)']<(datastat.loc[1,'Time at 4.15V (s)'])*2]
    dataframe=dataframe.loc[dataframe['Time constant current (s)']<(datastat.loc[1,'Time constant current (s)'])*2]
    dataframe=dataframe.loc[dataframe['Charging time (s)']<(datastat.loc[1,'Charging time (s)'])*2]
    
    #Affichage et enregistrement clean data
    print('Taille finale:',dataframe.shape)
    dataframe.to_csv(output_clean_data_path)
    
    #Affiche les statistiques du fichier cleaned
    cleaned_stat=dataframe.describe()
    print(cleaned_stat)
    cleaned_stat.to_csv(output_cleaned_datastat_path)
    
    
clean_data(dataframe,output_clean_data_path,output_init_datastat_path,output_cleaned_datastat_path)