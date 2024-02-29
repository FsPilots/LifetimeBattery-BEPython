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
    print(dataframe.head(),'\n')
    
    #Affiche le nb de données
    print('Taille initiale:',dataframe.shape)
    
    #Suppression des lignes avec des valeurs manquantes
    dataframe=dataframe.dropna(axis=0)
    
    #Affiche le nb de données après effactement des lignes vides
    print('Taille after supr na:',dataframe.shape,'\n')
    
    #Affiche les premières statistiques
    init_stat=dataframe.describe()
    print(init_stat,'\n')
    init_stat.to_csv(output_init_datastat_path)
    
    #Nettoyage
    
    # Suppression des lignes contenant des valeurs négatives dans n'importe quelle colonne
    dataframe = dataframe.loc[(dataframe >= 0).all(axis=1)]
    
    #Affiche le nb de données après effactement des lignes avec valeurs négatives
    print('Taille after supr neg:',dataframe.shape,'\n')

    # Charger le fichier CSV init_stat
    datastat = pd.read_csv(output_init_datastat_path)
    
    #Affiche les premières valeurs pour vérifier la bonne lecture du fichier
    print(datastat.head(),'\n')
    
    print('Moyenne des parametres initiaux:')
    print('Discharge Time (s):',datastat.loc[1,'Discharge Time (s)'])
    print('Decrement 3.6-3.4V (s):',datastat.loc[1,'Decrement 3.6-3.4V (s)'])
    print('Time at 4.15V (s):',datastat.loc[1,'Time at 4.15V (s)'])
    print('Time constant current (s):',datastat.loc[1,'Time constant current (s)'])
    print('Charging time (s):',datastat.loc[1,'Charging time (s)'],'\n')

    #Nettoyage approfondis du dataframe
    
    #Suppression des lignes avec des erreurs (basées sur la vérification par la valeur moyenne x 2 eafin d'exclure les valeurs trop grandes)
    #Concerne les colonnes Discharge Time (s)//Decrement 3.6-3.4V (s)//Time at 4.15V (s)//Time constant current (s)//Charging time (s)
    dataframe=dataframe.loc[dataframe['Discharge Time (s)']<(datastat.loc[1,'Discharge Time (s)'])*2]
    dataframe=dataframe.loc[dataframe['Decrement 3.6-3.4V (s)']<(datastat.loc[1,'Decrement 3.6-3.4V (s)'])*2]
    dataframe=dataframe.loc[dataframe['Time at 4.15V (s)']<(datastat.loc[1,'Time at 4.15V (s)'])*2]
    dataframe=dataframe.loc[dataframe['Time constant current (s)']<(datastat.loc[1,'Time constant current (s)'])*2]
    dataframe=dataframe.loc[dataframe['Charging time (s)']<(datastat.loc[1,'Charging time (s)'])*2]
    
    #Suppression des lignes avec des erreurs (basées sur la vérification par la valeur 25% /2 afin d'exclure les valeurs bien trop petites)
    #Concerne les colonnes Discharge Time (s)//Decrement 3.6-3.4V (s)//Time at 4.15V (s)//Time constant current (s)//Charging time (s)
    dataframe=dataframe.loc[dataframe['Discharge Time (s)']>(datastat.loc[4,'Discharge Time (s)'])/2]
    dataframe=dataframe.loc[dataframe['Decrement 3.6-3.4V (s)']>(datastat.loc[4,'Decrement 3.6-3.4V (s)'])/2]
    dataframe=dataframe.loc[dataframe['Time at 4.15V (s)']>(datastat.loc[4,'Time at 4.15V (s)'])/2]
    dataframe=dataframe.loc[dataframe['Time constant current (s)']>(datastat.loc[4,'Time constant current (s)'])/2]
    dataframe=dataframe.loc[dataframe['Charging time (s)']>(datastat.loc[4,'Charging time (s)'])/2]
    
    #Sauvegarde intermediaire du fichier cleaned
    temp_cleaned_stat=dataframe.descibe()
    print(temp_cleaned_stat,'\n')
    temp_cleaned_stat.to_csv('data/temp_data/temp_cleaned_data_stat')
    
    #Indication taille avant retouche finales
    print('Taille intermediaire:',dataframe.shape,'\n')
    
    #Retouches finales
    print('Retouche finales:')
    
    #Indicateur
    Runtime=0
    
    #Suppression des X% de valeurs les plus grandes et X% les plus petites afin d'affiner les resultats
    #Definition du paramètre X en %
    X=5
        
    while(Runtime==0 or (temp_cleaned_stat.loc[3,'Cycle_Index']!= init_stat.loc[3,'Cycle_Index'] and temp_cleaned_stat.loc[7,'Cycle_Index']!= init_stat.loc[7,'Cycle_Index'])):
        
        #Affichage des parametre pour vérification
        print('Valeur Runtime:',Runtime)
        print('Valeur parametre X:',X)
        print('Valeur Cycle Index ref: Min:',init_stat.loc[3,'Cycle_Index'],' Max:',init_stat.loc[7,'Cycle_Index'])
        print('Valeur Cycle Index Max:',temp_cleaned_stat.loc[3,'Cycle_Index'])
        print('Valeur Cycle Index Mix:',temp_cleaned_stat.loc[3,'Cycle_Index'],'\n')
        
        # Calculer le nombre de lignes à supprimer (X% de la taille du DataFrame)
        nb_lignes_a_supprimer = int(len(dataframe) * X/100)

        # Trier la colonne 'valeur' par ordre croissant
        dataframe_sorted = dataframe.sort_values(by='Discharge Time (s)')

        # Supprimer les X% les plus élevées et les X% les plus basses
        temp_dataframe = dataframe_sorted.iloc[nb_lignes_a_supprimer:-nb_lignes_a_supprimer]
        
        temp_cleaned_stat=temp_dataframe.descibe()
        temp_cleaned_stat.to_csv('data/temp_data/temp_cleaned_data_stat')
        
        Runtime=Runtime+1
        X=X/2
    
    
    #Enregistrement clean data
    dataframe=temp_dataframe
    
    #Affiche les statistiques du fichier cleaned
    cleaned_stat=dataframe.describe()
    print(cleaned_stat,'\n')
    cleaned_stat.to_csv(output_cleaned_datastat_path)
    
    print('Taille finale:',dataframe.shape,'\n')
    dataframe.to_csv(output_clean_data_path)
    
    
clean_data(dataframe,output_clean_data_path,output_init_datastat_path,output_cleaned_datastat_path)