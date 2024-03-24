import pandas as pd
import numpy as np

#Définir un chemin de sortie des statistiques finales
init_result_csv='data/temp_data/temp_result_init.csv'

def analyse_result(result_reg,result_svm,result_gau,prior_result,data_path):
    print('.....Analyse Resultats.....')
    
    #Lecture d'un tableau initial afin d'ecrire dedans
    df = pd.read_csv(init_result_csv)
    
    #Initialisation des valeurs et du tableau final des résultas
    result=-1
    time=-1
    indice_perfo=[0,0,0]
    
    #Pour chaque modèle, on vérifie qu'il a été entrainé et que les résultats sont valides
    #Puis, calcul d'un coef dépendant de la précision du modèle et de son temps d'entrainement
    #Ce coef dépend également de l'importance entre l'accuracy et le temps d'entrainement (Paramètre prior_result)
    for i in range(0,3):
        if(i==0):
            result=result_reg[0]
            time=result_reg[1]
        elif(i==1):
            result=result_svm[0]
            time=result_svm[1]
        elif(i==2):
            result=result_gau[0]
            time=result_gau[1]
        if(result<0 or time<0 or result>1):
            print('Resultat',i+1,'inutilisable')
            i=i+1
        else:
            print('Resultats modele',i+1,':',result)
            print('Temps entrainement modele',i+1,':',time)
            df.at[i,'Test_Accuracy']=result
            df.at[i,'Training_time']=time
            if(prior_result<100 and prior_result>0):
                indice_perfo[i]=(result+(1-prior_result/100)/time)
                print('Indice de performance du modele',i+1,':',indice_perfo[i])
                df.at[i,'Coef_Perfo']=indice_perfo[i]
            elif(prior_result==100):
                indice_perfo[i]=result
                df.at[i,'Coef_Perfo']=indice_perfo[i]
            elif(prior_result==0):
                indice_perfo[i]=time
                df.at[i,'Coef_Perfo']=indice_perfo[i]
    print('\n.....Tableau Resultats Finaux.....')
    print(df)
    df.to_csv(data_path)