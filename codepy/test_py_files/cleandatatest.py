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
def clean_data(dataframe,output_clean_data_path):
    
    #Affiche les premières valeurs pour vérifier la bonne lecture du fichier
    print('Affichage valeurs pour confirmer la bonne lecture:')
    print(dataframe.head(),'\n')
    
    #Affiche le nb de données
    init_shape=dataframe.shape
    print('Taille initiale:',init_shape)
    
    #Suppression des lignes avec des valeurs manquantes
    dataframe=dataframe.dropna(axis=0)
    
    #Affiche le nb de données après effactement des lignes vides
    print('Taille after supr na:',dataframe.shape,'\n')
    
    #Affiche les premières statistiques
    init_stat=dataframe.describe()
    print('Statistiques initiales:')
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
    #print('Affichage valeurs pour confirmer la bonne lecture:')
    #print(datastat.head(),'\n')
    
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
    
    #Indication taille avant retouche finales
    intermediaire_shape=dataframe.shape
    #print('Taille intermediaire:',intermediaire_shape,'\n')
    
    #Sauvegarde intermediaire du fichier cleaned
    temp_dataframe=dataframe
    temp_dataframe.to_csv('data/temp_data/temp_cleaned_dataframe.csv')
    temp_cleaned_data_stat=temp_dataframe.describe()
    #print('Statistiques intermediaires:')
    #print(temp_cleaned_data_stat,'\n')
    temp_cleaned_data_stat.to_csv('data/temp_data/temp_cleaned_data_stat.csv')
    
    #Retouches finales
    #print('Retouche finales:\n')
    
    #Indicateur
    Runtime=0
    Reduction_factor=0
    
    #Suppression des X% de valeurs les plus grandes et X% les plus petites afin d'affiner les resultats
    #Definition du paramètre X en %
    #Le but de cette definition est de retirer au maximum 50 valeurs donc si il y a 1000 données, X=5000/1000=5% soit 50 valeurs
    X=5000/(init_shape[0])
            
    while(Runtime<=1000):
        
        #Chargement des parametres temporaires
        temp_cleaned_data_stat=pd.read_csv('data/temp_data/temp_cleaned_data_stat.csv')
        
        # Calculer le nombre de lignes à supprimer (X% de la taille du DataFrame)
        nb_lignes_a_supprimer = int(len(dataframe) * X/100)

        # Trier la colonne 'valeur' par ordre croissant
        dataframe_sorted = dataframe.sort_values(by='Discharge Time (s)')

        # Supprimer les X% les plus élevées et les X% les plus basses
        temp_dataframe = dataframe_sorted.iloc[nb_lignes_a_supprimer:-nb_lignes_a_supprimer]
        
        # Trier la colonne 'valeur' par ordre croissant
        #dataframe_sorted = temp_dataframe.sort_values(by='Discharge Time (s)')
        #print(dataframe_sorted)
        
        temp_data_stat=temp_dataframe.describe()
        temp_data_stat.to_csv('data/temp_data/temp_data_stat.csv')
        temp_data_stat=pd.read_csv('data/temp_data/temp_data_stat.csv')
        
        #Affichage des parametre pour vérification
        #print('Valeur Runtime:',Runtime)
        #print('Valeur parametre X:',X,'%')
        #print('Valeur Cycle Index Before Last Clean: Min:',temp_cleaned_data_stat.loc[3,'Cycle_Index'],'Max:',temp_cleaned_data_stat.loc[7,'Cycle_Index'])
        #print('Valeur Cycle Index After',X,'% Clean: Min:',temp_data_stat.loc[3,'Cycle_Index'],'Max:',temp_data_stat.loc[7,'Cycle_Index'],'\n')
        
        if((temp_data_stat.loc[3,'Cycle_Index'] == temp_cleaned_data_stat.loc[3,'Cycle_Index']) and (temp_data_stat.loc[7,'Cycle_Index'] == temp_cleaned_data_stat.loc[7,'Cycle_Index'])):
            #Si la verification passe dès le premier tour, il se peut que des valeurs soient enlever en trop
            if(Runtime==0):
                #print("Reduction factor verif")    
                Runtime=0
                Reduction_factor=Reduction_factor+1
                #La valeur de X diminue en fonction du nombre de fois où la fonction est vérifier directement
                X=X*(intermediaire_shape[0]/init_shape[0])-X*Reduction_factor/1000
            else:
                Runtime=1001
                temp_data_stat.to_csv('data/temp_data/temp_data_stat.csv')
                #Enregistrement clean data
                dataframe=temp_dataframe
            
                
        else:
            Runtime=Runtime+1
            #La valeur de X diminue en fonction du nombre de valeurs qui ont déja été amputé à la base de donnée, la diminution de X s'accélère avec le temps
            X=X*(intermediaire_shape[0]/init_shape[0])-X*Runtime/1000
            
            #Quand la valeur de X est bien trop petite, il ne sert plus a rien de rester dans la boucle
            if(X<=(100/intermediaire_shape[0])):
                Runtime=1001
    
    
    #Affiche les statistiques du fichier cleaned
    cleaned_stat=dataframe.describe()
    print('Statistiques finales:')
    print(cleaned_stat,'\n')
    cleaned_stat.to_csv(output_cleaned_datastat_path)
    
    final_shape=dataframe.shape
    
    print('Rappel tailles:')
    print('Taille initiale:',init_shape)
    print('Moins:',init_shape[0]-intermediaire_shape[0])
    print('Taille intermediaire:',intermediaire_shape)
    print('Moins:',intermediaire_shape[0]-final_shape[0])
    print('Taille finale:',final_shape,'\n')
    print('Total suppr:',init_shape[0]-final_shape[0])
    print('% suppr:',round(100*((init_shape[0]-final_shape[0])/init_shape[0]),3),'%\n')
    dataframe.to_csv(output_clean_data_path)
       