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

#La variable choix modele permet de séléctionner le modèle a entrainer.
# 1 - Modèle Regression Lineaire
# 2 - Modèle SVM
# 3 - Modèle Processus Gaussien
# 4 - Entrainer les 3 modèles

def main(choix_fe,choix_model):
    #Etape 1: Nettoyage des données
    cd.clean_data(dataframe,output_clean_data_path)

    #Etape 2: Feature Engineering
    if(choix_fe==1):
        fe.feature_eng(output_clean_data_path,output_usable_data_path)
    else:
        print('.....Visualisation du feature engineering non affichee.....\n')

    #Etape 3: Entrainement modèle IA et Analyse resultats
    if(choix_model==1):
        RLmod.reglineaire(output_usable_data_path)
    elif(choix_model==2):
        RLmod.reglineaire(output_usable_data_path)
    elif(choix_model==3):
        RLmod.reglineaire(output_usable_data_path)
    elif(choix_model==4):
        RLmod.reglineaire(output_usable_data_path)
    else:
        print('.....Aucun modèle choisi.....')
    

main(0,1)