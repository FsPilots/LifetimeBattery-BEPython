import pandas as pd
import cleandata as cd
import feature_eng as fe
import modeldata_Reg_Linear as RLmod
import modeledata_SVM as svm
import modeldata_ProcesGaussienkrieging as Gau
import analyse_result as ar

# Charger le fichier CSV dans un DataFrame
dataframe = pd.read_csv('rendufinal/data/Battery_RUL.csv')

#Définir un chemin de sortie des données clean
output_clean_data_path='rendufinal/data/Battery_RUL_clean_data.csv'

#Définir un chemin de sortie des données utiliables par l'IA
output_usable_data_path='rendufinal/data/Battery_RUL_usable_data.csv'

#Définir un chemin de sortie des données des resultats des modèles d'IA
output_result_data_path='rendufinal/data/Battery_RUL_result.csv'

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
    
    #Définition variables de resultats (Resultats, Temps d'entrainement)
    result_Reg_Linear=(-1,-1)
    result_SVM=(-1,-1)
    result_Proces_Gaussien=(-1,-1)
    
    if(choix_model==1):
        result_Reg_Linear=RLmod.reglineaire(output_usable_data_path)
    elif(choix_model==2):
        result_SVM=svm.modelregressionGPR(output_usable_data_path)
    elif(choix_model==3):
        result_Proces_Gaussien=Gau.modelregressionGPR(output_usable_data_path)
    elif(choix_model==4):
        result_Reg_Linear=RLmod.reglineaire(output_usable_data_path)
        result_SVM=svm.modelregressionGPR(output_usable_data_path)
        result_Proces_Gaussien=Gau.modelregressionGPR(output_usable_data_path)
    else:
        print('.....Aucun modèle choisi.....')
        
    #Etape 4: Analyse des resultats
    #La variable priority result permet de définir l'importance de la précision du modèle par rapport à son temps d'entrainement.
    #Exprimé en %
    priority_result=80
    ar.analyse_result(result_Reg_Linear,result_SVM,result_Proces_Gaussien,priority_result,output_result_data_path)
    
#Programme
#main(Choix affichage FE; Choix execution modèle IA)
#main(1,4) est a lancé pour activer l'ensemble du programme et les 3 modèles 
#Attention, temps de compilation du Modlèle Proces Gaussien Long
main(1,4)